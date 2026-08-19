"""What the trainer reported about itself, kept instead of only printed.

mlx-lm's trainer already computes a loss, a learning rate, a rate in
iterations and tokens per second, a running supervised token count and a
peak memory figure, at its own reporting interval. It prints them and drops
them. Nothing that runs the trainer from outside can read them afterwards,
which is why the receipt used to declare the loss curve out of reach.

It is not out of reach. The trainer takes a callback and calls it with those
exact numbers, so this module is a callback that keeps them. Reading the
printed lines back out of stdout would have worked too and is the wrong
shape: a number parsed out of a log is a number whose meaning depends on the
format that printed it, and this repository's rule is that evidence travels
as objects.

Two of the recorded fields matter beyond curiosity. `trained_tokens` is the
running total of supervised tokens, because the trainer counts a batch's
tokens as the sum of the loss mask rather than its length; the end-to-end
measurement's fairness rule requires the arms to have trained on the same
supervised tokens, and this is the number that says whether they did. And
the timing fields are the trainer's own, so an arm that was slower says so
in its own words rather than only in the harness's clock.

What is recorded is what mlx-lm reported. These numbers are not recomputed
or checked here, and a receipt carrying them is a record of what the trainer
said, not a second opinion about whether it was right.

The class deliberately does not inherit from mlx-lm's `TrainingCallback`.
The trainer calls two methods on whatever it is given and never asks what
type it is, so inheriting would buy nothing and would drag a heavyweight
import into a module that otherwise needs none. It follows the same
`wrapped_callback` chaining convention mlx-lm's own callbacks use, so a user
who asked for wandb still gets wandb.
"""

from __future__ import annotations


class LossRecorder:
    """Keeps every progress report the trainer makes, and passes it on."""

    def __init__(self, wrapped=None):
        self.wrapped = wrapped
        self.train: list[dict] = []
        self.val: list[dict] = []

    def chained(self, wrapped):
        """Sit in front of the callback mlx-lm built, if it built one."""
        self.wrapped = wrapped
        return self

    def on_train_loss_report(self, train_info: dict) -> None:
        self.train.append(dict(train_info))
        if self.wrapped is not None:
            self.wrapped.on_train_loss_report(train_info)

    def on_val_loss_report(self, val_info: dict) -> None:
        self.val.append(dict(val_info))
        if self.wrapped is not None:
            self.wrapped.on_val_loss_report(val_info)

    def summary(self) -> dict:
        """The curve, plus the few readings worth having without arithmetic.

        The points are kept in full rather than reduced away: there is one
        per reporting interval, so the whole curve of a long run is still
        small, and a summary that discarded it would force the next question
        to be answered by re-running the training.
        """
        losses = [point["train_loss"] for point in self.train]
        return {
            "_meaning": "mlx-lm's own progress reports, recorded verbatim. "
                        "Not recomputed and not independently checked.",
            "train_reports": len(self.train),
            "val_reports": len(self.val),
            "first_train_loss": losses[0] if losses else None,
            "last_train_loss": losses[-1] if losses else None,
            "min_train_loss": min(losses) if losses else None,
            "last_val_loss": self.val[-1]["val_loss"] if self.val else None,
            # Supervised tokens, the quantity the end-to-end fairness rule
            # compares between arms. None when the trainer never reported.
            "trained_tokens": (self.train[-1]["trained_tokens"]
                               if self.train else None),
            "train_points": self.train,
            "val_points": self.val,
        }
