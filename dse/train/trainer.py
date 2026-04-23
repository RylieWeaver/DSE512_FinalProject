
# General
import json
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm

# Torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# DSE 512
from dse.model import TransformerConfig, MLMTransformer, SequenceRegressionTransformer
from dse.distributed import ParallelState, is_rank0, rank0_print, rank0_write, reduce_scalar, unwrap_model, resolve_device, broadcast_tensor
from dse.data import move_to
from dse.utils import Config
from .utils import init_optimizer_and_scheduler



class MLMTrainerConfig(Config):
    def __init__(
            self,
            log_every: int = 1,
            eval_every: int = 100,
            eval_batches: int = 10,
            batches_per_step: int = 1,
            learning_rate: float = 3e-5,
            warmup_steps: int = 0,
            decay_steps: Optional[int] = None,
            decay_type: str = "cosine",
            weight_decay: float = 1e-4,
            log_dir: Optional[Union[Path, str]] = None,
            checkpoint_dir: Optional[Union[Path, str]] = None,
            save_every: Optional[int] = None,
            amp_dtype: Optional[str] = "bfloat16",
            amp_enabled: bool = False,
            **kwargs
    ):
        # Read args
        self.log_every = log_every
        self.eval_every = eval_every
        self.eval_batches = eval_batches
        self.batches_per_step = batches_per_step
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self.weight_decay = weight_decay
        self.log_dir = Path(log_dir) if log_dir else Path("log")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir and is_rank0():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.amp_dtype = amp_dtype
        self.amp_enabled = amp_enabled

    @staticmethod
    def load(path: Union[Path, str]) -> "MLMTrainerConfig":
        path = Path(path) if path else None
        with path.open("r") as f:
            cfg = json.load(f)
        return MLMTrainerConfig(**cfg)


class MLMTrainer:
    def __init__(self, config, model, device: Optional[Union[torch.device, str]] = None, parallel_state: Optional[ParallelState] = None):
        # Read args
        self.cfg = config
        self.parallel_state = parallel_state if parallel_state else ParallelState()
        self.device = resolve_device(device, self.parallel_state)

        # Init objects
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)  # Sum reduction since we'll average manually to account for parallelism
        self.descriptors = ["Train w/ Grad", "Train w/o Grad", "Eval", "Test"]
        self.cumulative_metrics = {}
        self._init_cumulative_metrics(self.descriptors)

        # Init trainer state
        self.last_step = 0
        self.best_val_loss = float("inf")    

    def _init_optimizer(self):
        self.optimizer, self.scheduler = init_optimizer_and_scheduler(
            model=self.model,
            learning_rate=self.cfg.learning_rate,
            warmup_steps=self.cfg.warmup_steps,
            decay_steps=self.cfg.decay_steps,
            decay_type=self.cfg.decay_type,
            weight_decay=self.cfg.weight_decay,
        )

    def _init_cumulative_metrics(self, descriptors: list[str] | str):
        """
        These are used to accumulate metrics over multiple batches before averaging and logging.
        """
        if isinstance(descriptors, str):
            descriptors = [descriptors]
        for desc in descriptors:
            self.cumulative_metrics[desc] = {
                "loss": 0.0,
                "correct": 0,
                "count": 0
            }

    def _log_metrics(self, desc: str = ""):
        loss = self.cumulative_metrics[desc]["loss"] / self.cumulative_metrics[desc]["count"] if self.cumulative_metrics[desc]["count"] > 0 else 0.0
        acc = self.cumulative_metrics[desc]["correct"] / self.cumulative_metrics[desc]["count"] if self.cumulative_metrics[desc]["count"] > 0 else 0.0
        message = (
            f"[Step] {self.last_step}: "
            f"{desc} Loss: {loss:.4f}, "
            f"{desc} Accuracy: {acc:.4f}"
        )
        rank0_write(self.cfg.log_dir / "log.txt", message)
        rank0_print(message)

    def set_loaders(self, train_loader, val_loader, test_loader):
        self.train_loader = iter(train_loader)
        self.val_loader = iter(val_loader)
        self.test_loader = iter(test_loader)
        self.cfg.train_batch_size = train_loader.batch_size
        self.cfg.val_batch_size = val_loader.batch_size
        self.cfg.test_batch_size = test_loader.batch_size

    def _run_batch(self, token_ids, labels):
        # Broadcast ids/labels across sp group if sp parallel
        if self.parallel_state.sp_size > 1:
            sp_group = self.parallel_state.sp_group
            sp_src = dist.get_global_rank(sp_group, 0)
            is_src = is_rank0(sp_group)
            token_ids = broadcast_tensor(
                x=token_ids if is_src else None,
                group=sp_group,
                device=self.device,
                src=sp_src,
                dtype=torch.long,
            )
            labels = broadcast_tensor(
                x=labels if is_src else None,
                group=sp_group,
                device=self.device,
                src=sp_src,
                dtype=torch.long,
            )
        logits, labels = self.model(token_ids, labels)
        return logits, labels

    def _shape_data(self, logits, labels):
        """
        Reshape data for cross_entropy loss.

        logits: [B, S, V]  -->  [B*S, V]
        labels: [B, S]     -->  [B*S]
        """
        B, S, V = logits.size()
        logits = logits.reshape(B * S, V)
        labels = labels.reshape(B * S)
        return logits, labels

    def _accuracy_counter(self, logits, labels, ignore_index=-100, dim=-1):
        """
        The purpose of this function is to compute the number of correct predictions
        and the total count of predictions for accuracy calculation.

        The plumbing with ignore index is essentially there to ignore pads when 
        counting metrics (which shouldn't be there anyways).
        """
        predicted_class = torch.argmax(logits, dim=dim)
        mask = labels != ignore_index
        count = mask.sum().item()
        correct = (predicted_class[mask] == labels[mask]).sum().item()
        return correct, count
    
    def true_local_loss(self, local_loss, local_count):
        """
        Adjust loss scale to take into account true average reduction
        across parallel processes. This is only really necessary to do
        during grad steps, since eval metrics can keep a running sum and
        average.
        """
        grad_group = self.parallel_state.grad_sync_group if self.parallel_state.grad_sync_group else self.parallel_state.world_group
        total_count = reduce_scalar(
            local_count, device=self.device, 
            group=grad_group, average=False
        )
        grad_sync_size = dist.get_world_size(grad_group) if grad_group else self.parallel_state.world_size
        # NOTE: No multiplication by local count here because our
        # criterion has a "sum" reduction, not a "mean" reduction.
        scale = grad_sync_size * (1 / total_count) if local_count > 0 else 1.0
        true_local_loss = local_loss * scale
        return true_local_loss

    def _compute_metrics(self, logits, labels):
        # Reshape for loss/accuracy computation
        logits, labels = self._shape_data(logits, labels)
        # Loss computation
        loss = self.criterion(logits, labels)
        # Accuracy computation
        correct, count = self._accuracy_counter(logits, labels)
        return loss, correct, count

    def _inc_metrics(self, loss, correct, count, desc):
        self.cumulative_metrics[desc]["loss"] += loss.item() if isinstance(loss, torch.Tensor) else loss
        self.cumulative_metrics[desc]["correct"] += correct.item() if isinstance(correct, torch.Tensor) else correct
        self.cumulative_metrics[desc]["count"] += count.item() if isinstance(count, torch.Tensor) else count

    def _reduce_metrics(self, desc):
        """
        This reduces metrics over sequence and data parallelism.

        Note that either/both parallelism dimensions may be size 1, 
        in which case the reduction is a no-op for that dimension.

        Note that there's no need to average over batches here
        because we keep a running sum of count to average over.
        """
        self.cumulative_metrics[desc]["loss"] = reduce_scalar(
            self.cumulative_metrics[desc]["loss"], device=self.device, 
            group=self.parallel_state.world_group, average=False
        )
        self.cumulative_metrics[desc]["correct"] = reduce_scalar(
            self.cumulative_metrics[desc]["correct"], device=self.device, 
            group=self.parallel_state.world_group, average=False
        )
        self.cumulative_metrics[desc]["count"] = reduce_scalar(
            self.cumulative_metrics[desc]["count"], device=self.device, 
            group=self.parallel_state.world_group, average=False
        )
    
    def _loop_without_grad(self, loader, steps):
        total_loss, total_correct, total_count = 0.0, 0, 0
        with torch.no_grad():
            for _ in range(steps):
                token_ids, labels = next(loader)
                token_ids, labels = move_to(token_ids, self.device), move_to(labels, self.device)
                with torch.autocast(device_type=self.device.type, dtype=getattr(torch, self.cfg.amp_dtype), enabled=self.cfg.amp_enabled):
                    logits, labels = self._run_batch(token_ids, labels)
                    batch_loss, batch_correct, batch_count = self._compute_metrics(logits, labels)
                total_loss += batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss
                total_correct += batch_correct.item() if isinstance(batch_correct, torch.Tensor) else batch_correct
                total_count += batch_count.item() if isinstance(batch_count, torch.Tensor) else batch_count
        return total_loss, total_correct, total_count    

    def _run_single_eval(self, loader, desc):
        # Setup
        self._init_cumulative_metrics([desc])  # Reset metrics

        # Collect and avg
        loss, correct, count = self._loop_without_grad(loader, self.cfg.eval_batches)
        self._inc_metrics(loss, correct, count, desc)
        self._reduce_metrics(desc)  # Reduce metrics over parallel processes
        
        # Log and reset
        self._log_metrics(desc)

    def _run_eval(self):
        self.model.eval()
        self._run_single_eval(self.train_loader, self.descriptors[1])  # Train w/o Grad
        self._run_single_eval(self.val_loader, self.descriptors[2])    # Eval
        self._run_single_eval(self.test_loader, self.descriptors[3])   # Test

    def train(self, steps=10000, end_step=None):
        # Setup
        if end_step is None:
            end_step = self.last_step + steps

        # Initial evaluation before training
        if self.last_step == 0:
            self._run_eval()
            self.model.train()  # Switch back to train

        # Training loop
        while self.last_step < end_step:
            self.optimizer.zero_grad(set_to_none=True)
            # Accumulate gradients
            for batch_idx in range(self.cfg.batches_per_step):
                # Forward
                token_ids, labels = next(self.train_loader)     # [B, S] (int)
                token_ids, labels = move_to(token_ids, self.device), move_to(labels, self.device)
                with torch.autocast(device_type=self.device.type, dtype=getattr(torch, self.cfg.amp_dtype), enabled=self.cfg.amp_enabled):
                    logits, labels = self._run_batch(token_ids, labels)   # [B, S, V], [B, S]
                    # Loss and backward
                    loss, correct, count = self._compute_metrics(logits, labels)
                self._inc_metrics(loss, correct, count, self.descriptors[0])
                # Properly scale the loss for grad calculation to account for parallelism and accumulation
                grad_loss = self.true_local_loss(loss, count) / self.cfg.batches_per_step
                if batch_idx < self.cfg.batches_per_step - 1 and isinstance(self.model, DDP):
                    with self.model.no_sync():
                        grad_loss.backward()
                else:
                    grad_loss.backward()
            # Optimizer step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.last_step += 1

            # Reached log step
            if self.last_step % self.cfg.log_every == 0:
                d = self.descriptors[0]  # Train descriptor for easy reference
                self._reduce_metrics(d)  # Reduce metrics over parallel processes
                self._log_metrics(desc=d)
                self._init_cumulative_metrics(d)  # Reset metrics

            # Reached eval step
            if self.last_step % self.cfg.eval_every == 0:
                self._run_eval()
                val_num = self.cumulative_metrics[self.descriptors[2]]["loss"]
                val_den = self.cumulative_metrics[self.descriptors[2]]["count"]
                val_loss = val_num / val_den if val_den > 0 else float("inf")
                if val_loss < self.best_val_loss:  # Check if best val loss
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")
                self.model.train()  # Switch back to train

            # Reached checkpoint step
            if self.cfg.checkpoint_dir and self.cfg.save_every and self.last_step % self.cfg.save_every == 0:
                self.save_checkpoint(f"step_{self.last_step}")

    def _load_state_dict(self, path: Union[Path, str]):
        state_dict = torch.load(path)
        self.last_step = state_dict["step"]
        self.best_val_loss = state_dict.get("best_val_loss", float("inf"))

    def save_checkpoint(self, ckpt_dir: str):
        # Setup
        model = unwrap_model(self.model)  # Removes DDP wrapper if present
        save_dir = self.cfg.checkpoint_dir / ckpt_dir

        # Save (only on rank 0)
        if is_rank0():
            save_dir.mkdir(parents=True, exist_ok=True)
            # Model
            model.cfg.save(save_dir / "model_config.json")
            torch.save(model.state_dict(), save_dir / f"model.pt")
            # Trainer
            self.cfg.save(save_dir / "trainer_config.json")
            torch.save({"step": self.last_step, "best_val_loss": self.best_val_loss}, save_dir / "trainer.pt")
            # Optimizer / scheduler
            torch.save(self.optimizer.state_dict(), save_dir / "optimizer.pt")
            torch.save(self.scheduler.state_dict(), save_dir / "scheduler.pt")

    @staticmethod
    def load_checkpoint(dir: Union[Path, str], device: torch.device, parallel_state: ParallelState = None) -> "MLMTrainer":
        # Setup
        dir = Path(dir)
        parallel_state = parallel_state if parallel_state else ParallelState()

        # Model
        model_dict_path = dir / "model_config.json"
        with model_dict_path.open("r") as f:
            model_dict = json.load(f)
        # NOTE: We only support MLMTransformer here for simplicity
        model_cfg = TransformerConfig(**model_dict)
        model = MLMTransformer(model_cfg, parallel_state).to(device)
        state_dict = torch.load(dir / f"model.pt", weights_only=True, map_location=device)
        model.load_state_dict(state_dict)

        # Trainer
        trainer_dict_path = dir / "trainer_config.json"
        trainer_cfg = MLMTrainerConfig.load(trainer_dict_path)
        trainer = MLMTrainer(config=trainer_cfg, model=model, device=device, parallel_state=parallel_state)
        trainer._load_state_dict(dir / "trainer.pt")
        # Optimizer / scheduler
        trainer._init_optimizer()
        trainer.optimizer.load_state_dict(torch.load(dir / "optimizer.pt", map_location=device))
        scheduler_path = dir / "scheduler.pt"
        trainer.scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
        return trainer


class SequenceRegressionTrainerConfig(Config):
    def __init__(
            self,
            log_every: int = 1,
            eval_every: int = 1,
            batches_per_step: int = 1,
            learning_rate: float = 3e-5,
            warmup_steps: int = 0,
            decay_steps: Optional[int] = None,
            decay_type: str = "cosine",
            weight_decay: float = 1e-4,
            log_dir: Optional[Union[Path, str]] = None,
            checkpoint_dir: Optional[Union[Path, str]] = None,
            save_every: Optional[int] = None,
            amp_dtype: Optional[str] = "bfloat16",
            amp_enabled: bool = False,
            **kwargs
    ):
        # Read args
        self.log_every = log_every
        self.eval_every = eval_every
        self.batches_per_step = batches_per_step
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self.weight_decay = weight_decay
        self.log_dir = Path(log_dir) if log_dir else Path("log")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir and is_rank0():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.amp_dtype = amp_dtype
        self.amp_enabled = amp_enabled

    @staticmethod
    def load(path: Union[Path, str]) -> "SequenceRegressionTrainerConfig":
        path = Path(path) if path else None
        with path.open("r") as f:
            cfg = json.load(f)
        return SequenceRegressionTrainerConfig(**cfg)


class SequenceRegressionTrainer:
    def __init__(self, config, model, device: Optional[Union[torch.device, str]] = None, parallel_state: Optional[ParallelState] = None):
        # Read args
        self.cfg = config
        self.parallel_state = parallel_state if parallel_state else ParallelState()
        self.device = resolve_device(device, self.parallel_state)

        # Init objects
        self.model = model.to(self.device)
        self.criterion = torch.nn.MSELoss(reduction="sum")  # Sum reduction since we'll average manually to account for parallelism
        self.descriptors = ["Train w/ Grad", "Train w/o Grad", "Eval", "Test"]
        self.cumulative_metrics = {}
        self._init_cumulative_metrics(self.descriptors)

        # Init trainer state
        self.last_epoch = 0
        self.best_val_loss = float("inf")    

    def _init_optimizer(self):
        self.optimizer, self.scheduler = init_optimizer_and_scheduler(
            model=self.model,
            learning_rate=self.cfg.learning_rate,
            warmup_steps=self.cfg.warmup_steps,
            decay_steps=self.cfg.decay_steps,
            decay_type=self.cfg.decay_type,
            weight_decay=self.cfg.weight_decay,
        )

    def _init_cumulative_metrics(self, descriptors: list[str] | str):
        """
        These are used to accumulate metrics over multiple batches before averaging and logging.
        """
        if isinstance(descriptors, str):
            descriptors = [descriptors]
        for desc in descriptors:
            self.cumulative_metrics[desc] = {
                "loss": 0.0,
                "count": 0
            }

    def _log_metrics(self, desc: str = ""):
        loss = self.cumulative_metrics[desc]["loss"] / self.cumulative_metrics[desc]["count"] if self.cumulative_metrics[desc]["count"] > 0 else 0.0
        message = (
            f"[Epoch] {self.last_epoch}: "
            f"{desc} Loss: {loss:.4f}"

        )
        rank0_write(self.cfg.log_dir / "log.txt", message)
        rank0_print(message)

    def set_loaders(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg.train_batch_size = train_loader.batch_size
        self.cfg.val_batch_size = val_loader.batch_size
        self.cfg.test_batch_size = test_loader.batch_size

    def _run_batch(self, inputs, labels):
        # Broadcast ids/labels across sp group if sp parallel
        if self.parallel_state.sp_size > 1:
            sp_group = self.parallel_state.sp_group
            sp_src = dist.get_global_rank(sp_group, 0)
            is_src = is_rank0(sp_group)
            token_ids = inputs["token_ids"] if is_src else None
            temperatures = inputs["temperatures"] if is_src else None
            inputs = {
                "token_ids": broadcast_tensor(
                    x=token_ids,
                    group=sp_group,
                    device=self.device,
                    src=sp_src,
                    dtype=torch.long,
                ),
                "temperatures": broadcast_tensor(
                    x=temperatures,
                    group=sp_group,
                    device=self.device,
                    src=sp_src,
                    dtype=torch.float32,
                ),
            }
            labels = broadcast_tensor(
                x=labels if is_src else None,
                group=sp_group,
                device=self.device,
                src=sp_src,
                dtype=torch.float32,
            )
        preds, labels = self.model(inputs, labels)
        return preds, labels

    def _shape_data(self, preds, labels):
        """
        Reshape data for MSE loss.

        preds:  [?]  -->  [B, *]
        labels: [?]  -->  [B, *]
        """
        B = preds.size(0)
        preds = preds.reshape(B, -1)
        labels = labels.reshape(B, -1)
        return preds, labels
    
    def true_local_loss(self, local_loss, local_count):
        """
        Adjust loss scale to take into account true average reduction
        across parallel processes. This is only really necessary to do
        during grad steps, since eval metrics can keep a running sum and
        average.
        """
        grad_group = self.parallel_state.grad_sync_group if self.parallel_state.grad_sync_group else self.parallel_state.world_group
        total_count = reduce_scalar(
            local_count, device=self.device, 
            group=grad_group, average=False
        )
        grad_sync_size = dist.get_world_size(grad_group) if grad_group else self.parallel_state.world_size
        # NOTE: No multiplication by local count here because our
        # criterion has a "sum" reduction, not a "mean" reduction.
        scale = grad_sync_size * (1 / total_count) if local_count > 0 else 1.0
        true_local_loss = local_loss * scale
        return true_local_loss

    def _compute_metrics(self, preds, labels):
        # Reshape for preds computation
        preds, labels = self._shape_data(preds, labels)
        # Loss/count computation
        loss = self.criterion(preds, labels)
        count = labels.numel()
        return loss, count

    def _inc_metrics(self, loss, count, desc):
        self.cumulative_metrics[desc]["loss"] += loss.item() if isinstance(loss, torch.Tensor) else loss
        self.cumulative_metrics[desc]["count"] += count.item() if isinstance(count, torch.Tensor) else count

    def _reduce_metrics(self, desc):
        """
        This reduces metrics over sequence and data parallelism.

        Note that either/both parallelism dimensions may be size 1, 
        in which case the reduction is a no-op for that dimension.

        Note that there's no need to average over batches here
        because we keep a running sum of count to average over.
        """
        self.cumulative_metrics[desc]["loss"] = reduce_scalar(
            self.cumulative_metrics[desc]["loss"], device=self.device, 
            group=self.parallel_state.world_group, average=False
        )
        self.cumulative_metrics[desc]["count"] = reduce_scalar(
            self.cumulative_metrics[desc]["count"], device=self.device, 
            group=self.parallel_state.world_group, average=False
        )
    
    def _eval_epoch(self, loader):
        total_loss, total_count = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(loader, total=len(loader), desc="Evaluating", disable=not is_rank0(), leave=False):
                inputs, labels = move_to(inputs, self.device), move_to(labels, self.device)
                with torch.autocast(device_type=self.device.type, dtype=getattr(torch, self.cfg.amp_dtype), enabled=self.cfg.amp_enabled):
                    preds, labels = self._run_batch(inputs, labels)
                    batch_loss, batch_count = self._compute_metrics(preds, labels)
                total_loss += batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss
                total_count += batch_count.item() if isinstance(batch_count, torch.Tensor) else batch_count
        return total_loss, total_count

    def _run_single_eval(self, loader, desc):
        # Setup
        self._init_cumulative_metrics([desc])  # Reset metrics

        # Collect and avg
        loss, count = self._eval_epoch(loader)
        self._inc_metrics(loss, count, desc)
        self._reduce_metrics(desc)  # Reduce metrics over parallel processes
        
        # Log and reset
        self._log_metrics(desc)

    def _run_eval(self):
        self.model.eval()
        self._run_single_eval(self.train_loader, self.descriptors[1])  # Train w/o Grad
        self._run_single_eval(self.val_loader, self.descriptors[2])    # Eval
        self._run_single_eval(self.test_loader, self.descriptors[3])   # Test

    def _train_epoch(self):
        # Setup
        total_loss, total_count = 0.0, 0
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.last_epoch)
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, (inputs, labels) in enumerate(
            tqdm(
                self.train_loader,
                total=len(self.train_loader),
                desc="Training",
                disable=not is_rank0(),
                leave=False,
            )
        ):
            # Forward
            inputs, labels = move_to(inputs, self.device), move_to(labels, self.device)  # [B, S], [B, *]
            with torch.autocast(device_type=self.device.type, dtype=getattr(torch, self.cfg.amp_dtype), enabled=self.cfg.amp_enabled):
                preds, labels = self._run_batch(inputs, labels)  # [B, S, V], [B, S]
                # Loss
                loss, count = self._compute_metrics(preds, labels)
            # Backward (properly scale the loss for grad calculation to account for parallelism and accumulation)
            grad_loss = self.true_local_loss(loss, count) / self.cfg.batches_per_step
            if ((batch_idx + 1) % self.cfg.batches_per_step != 0) and isinstance(self.model, DDP):
                with self.model.no_sync():
                    grad_loss.backward()
            else:
                grad_loss.backward()
            # Increment
            total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            total_count += count.item() if isinstance(count, torch.Tensor) else count
            # Optimizer step
            if ((batch_idx + 1) % self.cfg.batches_per_step == 0) or (batch_idx == len(self.train_loader) - 1):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
        return total_loss, total_count

    def train(self, epochs=100, end_epoch=None):
        # Setup
        if end_epoch is None:
            end_epoch = self.last_epoch + epochs

        # Initial evaluation before training
        if self.last_epoch == 0:
            self._run_eval()
            self.model.train()  # Switch back to train

        # Training loop
        while self.last_epoch < end_epoch:
            # Train epoch
            train_loss, train_count = self._train_epoch()
            self._inc_metrics(train_loss, train_count, self.descriptors[0])
            self.last_epoch += 1

            # Reached log step
            if self.last_epoch % self.cfg.log_every == 0:
                d = self.descriptors[0]  # Train descriptor for easy reference
                self._reduce_metrics(d)  # Reduce metrics over parallel processes
                self._log_metrics(desc=d)
                self._init_cumulative_metrics(d)  # Reset metrics

            # Reached eval step
            if self.last_epoch % self.cfg.eval_every == 0:
                self._run_eval()
                val_num = self.cumulative_metrics[self.descriptors[2]]["loss"]
                val_den = self.cumulative_metrics[self.descriptors[2]]["count"]
                val_loss = val_num / val_den if val_den > 0 else float("inf")
                if val_loss < self.best_val_loss:  # Check if best val loss
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")
                self.model.train()  # Switch back to train

            # Reached checkpoint step
            if self.cfg.checkpoint_dir and self.cfg.save_every and self.last_epoch % self.cfg.save_every == 0:
                self.save_checkpoint(f"step_{self.last_epoch}")

    def _load_state_dict(self, path: Union[Path, str]):
        state_dict = torch.load(path)
        self.last_epoch = state_dict["epoch"]
        self.best_val_loss = state_dict.get("best_val_loss", float("inf"))

    def save_checkpoint(self, ckpt_dir: str):
        # Setup
        model = unwrap_model(self.model)  # Removes DDP wrapper if present
        save_dir = self.cfg.checkpoint_dir / ckpt_dir

        # Save (only on rank 0)
        ## NOTE: Sandwiching with barriers prevents timeout due to rank0 taking a long time to save
        if dist.is_initialized():
            dist.barrier()
        if is_rank0():
            save_dir.mkdir(parents=True, exist_ok=True)
            # Model
            model.cfg.save(save_dir / "model_config.json")
            torch.save(model.state_dict(), save_dir / f"model.pt")
            # Trainer
            self.cfg.save(save_dir / "trainer_config.json")
            torch.save({"epoch": self.last_epoch, "best_val_loss": self.best_val_loss}, save_dir / "trainer.pt")
            # Optimizer / scheduler
            torch.save(self.optimizer.state_dict(), save_dir / "optimizer.pt")
            torch.save(self.scheduler.state_dict(), save_dir / "scheduler.pt")
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def load_checkpoint(dir: Union[Path, str], device: torch.device, parallel_state: ParallelState = None) -> "SequenceRegressionTrainer":
        # Setup
        dir = Path(dir)
        parallel_state = parallel_state if parallel_state else ParallelState()

        # Model
        model_dict_path = dir / "model_config.json"
        with model_dict_path.open("r") as f:
            model_dict = json.load(f)
        # NOTE: We only support SequenceRegressionTransformer here for simplicity
        model_cfg = TransformerConfig(**model_dict)
        model = SequenceRegressionTransformer(model_cfg, parallel_state).to(device)
        state_dict = torch.load(dir / f"model.pt", weights_only=True, map_location=device)
        model.load_state_dict(state_dict)

        # Trainer
        trainer_dict_path = dir / "trainer_config.json"
        trainer_cfg = SequenceRegressionTrainerConfig.load(trainer_dict_path)
        trainer = SequenceRegressionTrainer(config=trainer_cfg, model=model, device=device, parallel_state=parallel_state)
        trainer._load_state_dict(dir / "trainer.pt")
        # Optimizer / scheduler
        trainer._init_optimizer()
        trainer.optimizer.load_state_dict(torch.load(dir / "optimizer.pt", map_location=device))
        scheduler_path = dir / "scheduler.pt"
        trainer.scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
        return trainer
