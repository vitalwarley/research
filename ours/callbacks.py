from lightning.pytorch.callbacks import Callback


class GradientInspectionCallback(Callback):
    def on_after_backward(self, trainer, pl_module):
        # Ensure we're in the training phase
        if trainer.state.fn == "fit":
            # Example: inspect gradient norms for potential conflict
            for name, param in pl_module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm()
                    trainer.logger.experiment.log({f"{name}_grad_norm": grad_norm, "global_step": trainer.global_step})
