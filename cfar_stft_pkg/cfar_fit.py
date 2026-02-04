from __future__ import annotations

import warnings

from .cfar2d import CFAR2D


def ensure_cfar_fits(self, n_freq: int, n_time: int) -> bool:
    """
    Extracted from CFARSTFTDetector._ensure_cfar_fits.

    Mutates:
      - self.cfar (may be re-created with smaller guard/train)
    """
    max_total_v = (n_freq - 1) // 2
    max_total_h = (n_time - 1) // 2

    if max_total_v < 1 or max_total_h < 1:
        warnings.warn(
            "STFT grid too small for CFAR window; skipping detection.",
            RuntimeWarning,
        )
        return False

    current_guard_v = self.cfar.N_G_v
    current_guard_h = self.cfar.N_G_h
    current_train_v = self.cfar.N_T_v
    current_train_h = self.cfar.N_T_h

    total_v = current_guard_v + current_train_v
    total_h = current_guard_h + current_train_h

    if total_v <= max_total_v and total_h <= max_total_h:
        return True

    # Reduce training cells first, then guard if needed
    new_total_v = min(total_v, max_total_v)
    new_total_h = min(total_h, max_total_h)

    new_train_v = max(0, new_total_v - current_guard_v)
    new_train_h = max(0, new_total_h - current_guard_h)

    if new_train_v == 0 and current_guard_v > new_total_v:
        new_guard_v = new_total_v
    else:
        new_guard_v = current_guard_v

    if new_train_h == 0 and current_guard_h > new_total_h:
        new_guard_h = new_total_h
    else:
        new_guard_h = current_guard_h

    # If still invalid, clamp guard to fit
    new_guard_v = min(new_guard_v, max_total_v)
    new_guard_h = min(new_guard_h, max_total_h)
    new_train_v = max(0, min(new_train_v, max_total_v - new_guard_v))
    new_train_h = max(0, min(new_train_h, max_total_h - new_guard_h))

    if (new_guard_v, new_train_v, new_guard_h, new_train_h) != (
        current_guard_v,
        current_train_v,
        current_guard_h,
        current_train_h,
    ):
        warnings.warn(
            "CFAR window resized to fit STFT grid: "
            f"guard_v={new_guard_v}, train_v={new_train_v}, "
            f"guard_h={new_guard_h}, train_h={new_train_h}.",
            RuntimeWarning,
        )
        self.cfar = CFAR2D(
            guard_cells_v=new_guard_v,
            guard_cells_h=new_guard_h,
            training_cells_v=new_train_v,
            training_cells_h=new_train_h,
            pfa=self.cfar.pfa,
            distribution=self.cfar.distribution,
        )

    return True

