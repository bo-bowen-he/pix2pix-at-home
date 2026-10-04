#!/usr/bin/env python3
"""Patch vendor/pix2pix-tensorflow/pix2pix.py so training can't NaN out.

The cause: the discriminator loss is written as log(1 - p + 1e-12). In
float32 the 1e-12 is lost, so the moment the discriminator outputs exactly
1.0 for a generated patch the loss is log(0) = inf and its gradient is NaN.
Unpatched, that NaN is written into the weights and training dies with
"Nan in summary histogram".

This patch:
  1. computes the same GAN losses from the discriminator's logits (softplus
     form), which is mathematically identical and cannot overflow;
  2. as a safety net, skips any step whose gradients are still non-finite and
     prints the running count as "skipped_steps" on each progress line.

Checkpoints stay compatible in both directions: no variables are added or
renamed.

    python scripts/apply_nan_guard.py          # apply (safe to re-run; upgrades v1)
    python scripts/apply_nan_guard.py --undo   # restore the original file
"""
import shutil
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "vendor/pix2pix-tensorflow/pix2pix.py"
BACKUP = TARGET.with_suffix(".py.pre_nan_guard")
MARKER = "# nan-guard v2"

EDITS = [
    # 1. gradient guard helper + one extra Model field
    (
        'gen_loss_L1, gen_grads_and_vars, train")\n',
        'gen_loss_L1, gen_grads_and_vars, train, step_ok")\n'
        "\n\n"
        "def _guard_grads(grads_and_vars):  " + MARKER + "\n"
        "    # if any gradient in this step is NaN/inf, replace the whole update with zeros\n"
        "    ok = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g, _ in grads_and_vars if g is not None])\n"
        "    guarded = [(g if g is None else tf.where(tf.fill(tf.shape(g), ok), g, tf.zeros_like(g)), v) for g, v in grads_and_vars]\n"
        "    return guarded, ok\n",
    ),
    # 2. the actual fix: same GAN losses, computed from the discriminator's
    #    logits. The original log(1 - p + 1e-12) is inf in float32 once p == 1.
    (
        "        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))\n",
        '        assert predict_real.op.type == "Sigmoid" and predict_fake.op.type == "Sigmoid"\n'
        "        logits_real = predict_real.op.inputs[0]\n"
        "        logits_fake = predict_fake.op.inputs[0]\n"
        "        discrim_loss = tf.reduce_mean(tf.nn.softplus(-logits_real) + tf.nn.softplus(logits_fake))\n",
    ),
    (
        "        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))\n",
        "        gen_loss_GAN = tf.reduce_mean(tf.nn.softplus(-logits_fake))\n",
    ),
    # 3. safety net: guard both optimisers
    (
        "        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)\n",
        "        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)\n"
        "        discrim_grads_and_vars, discrim_ok = _guard_grads(discrim_grads_and_vars)\n",
    ),
    (
        "            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)\n",
        "            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)\n"
        "            gen_grads_and_vars, gen_ok = _guard_grads(gen_grads_and_vars)\n",
    ),
    (
        "        train=tf.group(update_losses, incr_global_step, gen_train),\n",
        "        train=tf.group(update_losses, incr_global_step, gen_train),\n"
        "        step_ok=tf.logical_and(discrim_ok, gen_ok),\n",
    ),
    # 4. count and report skipped steps
    (
        "            start = time.time()\n\n            for step in range(max_steps):\n",
        "            start = time.time()\n            skipped_steps = 0\n\n            for step in range(max_steps):\n",
    ),
    (
        '                    "global_step": sv.global_step,\n                }\n',
        '                    "global_step": sv.global_step,\n                    "step_ok": model.step_ok,\n                }\n',
    ),
    (
        "                results = sess.run(fetches, options=options, run_metadata=run_metadata)\n",
        "                results = sess.run(fetches, options=options, run_metadata=run_metadata)\n"
        '                if not results["step_ok"]:\n'
        "                    skipped_steps += 1\n",
    ),
    (
        '                    print("gen_loss_L1", results["gen_loss_L1"])\n',
        '                    print("gen_loss_L1", results["gen_loss_L1"])\n'
        '                    print("skipped_steps", skipped_steps)\n',
    ),
]


def main():
    if not TARGET.exists():
        sys.exit(f"Not found: {TARGET} (run setup.sh first)")

    if "--undo" in sys.argv:
        if not BACKUP.exists():
            sys.exit("No backup found; nothing to undo.")
        shutil.copyfile(BACKUP, TARGET)
        BACKUP.unlink()
        print(f"Restored original {TARGET.name}")
        return

    src = TARGET.read_text()
    if MARKER in src:
        print("Already patched; nothing to do.")
        return
    if BACKUP.exists():
        # an earlier version of this patch is applied: start again from the original
        src = BACKUP.read_text()

    out = src
    for old, new in EDITS:
        if out.count(old) != 1:
            sys.exit(f"Patch anchor not found exactly once, file left untouched:\n{old!r}")
        out = out.replace(old, new)

    if not BACKUP.exists():
        shutil.copyfile(TARGET, BACKUP)
    TARGET.write_text(out)
    print(f"Patched {TARGET.name} (original saved as {BACKUP.name})")


if __name__ == "__main__":
    main()
