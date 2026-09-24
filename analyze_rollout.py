"""Analyze a MeshGraphNets rollout pickle: MSE-vs-horizon curve + GIF render.

The official `meshgraphnets/plot_cloth.py` only calls `plt.show(block=True)` --
it displays an interactive window and saves nothing. On a headless box that
means the rollout produces no artifact at all. This script is an *addition*, not
a modification: the official file is left byte-identical.

It produces the two things you actually want from a reproduction:

  1. `--mse_out`  : the paper's headline plot -- per-step MSE against the
                    ground-truth rollout, on a log scale, one line per
                    trajectory plus the mean. This is the figure that shows
                    whether the model stays stable out to 200 steps or diverges.
  2. `--gif_out`  : a rendered animation of pred vs. ground truth, saved with
                    Pillow (no ffmpeg on this host).

Usage:
    python analyze_rollout.py --rollout_path=.../rollout.pkl \
        --mse_out=.../mse_curve.png --gif_out=.../rollout.gif
"""

import os
import pickle

import matplotlib

matplotlib.use("Agg")  # headless: must be set before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("rollout_path", None, "Path to rollout pickle file")
flags.DEFINE_string("mse_out", None, "Where to write the MSE-vs-horizon PNG")
flags.DEFINE_string("gif_out", None, "Where to write the rendered GIF")
flags.DEFINE_integer("max_gif_frames", 120, "Cap on GIF frames (keeps file small)")
flags.DEFINE_integer("skip", 10, "Render every Nth step, matching plot_cloth.py")
flags.DEFINE_string(
    "batch_gif_dir", None,
    "Write enhanced compare + overlay GIFs for every trajectory to this directory")
flags.DEFINE_integer("gif_fps", 10, "Frame rate for rendered GIFs")

flags.mark_flag_as_required("rollout_path")


def _load(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def _is_cloth(traj):
    """Cloth rollouts carry 'world_pos' targets; CFD carries 'velocity'."""
    return "gt_pos" in traj


def plot_mse(trajectories, out_path):
    """Per-step MSE vs horizon, mirroring cloth_eval/cfd_eval's masked metric."""
    fig, ax = plt.subplots(figsize=(9, 6))

    curves = []
    for traj in trajectories:
        pred = traj["pred_pos"] if _is_cloth(traj) else traj["pred_velocity"]
        gt = traj["gt_pos"] if _is_cloth(traj) else traj["gt_velocity"]
        # Per-step, per-node MSE averaged over nodes and spatial dims.
        mse = np.mean((pred - gt) ** 2, axis=tuple(range(1, pred.ndim)))
        curves.append(mse)
        ax.plot(np.arange(len(mse)), mse, alpha=0.35, linewidth=1)

    max_len = max(len(c) for c in curves)
    stacked = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        stacked[i, : len(c)] = c
    mean_curve = np.nanmean(stacked, axis=0)

    ax.plot(np.arange(max_len), mean_curve, color="black", linewidth=2.5,
            label="mean over %d trajectories" % len(curves))

    # The horizons cloth_eval/cfd_eval report.
    for horizon in (1, 10, 20, 50, 100, 200):
        if horizon < max_len:
            ax.axvline(horizon, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.annotate("%d" % horizon, (horizon, ax.get_ylim()[1]),
                        fontsize=8, color="grey", ha="center", va="bottom")

    ax.set_yscale("log")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("MeshGraphNets rollout error vs. horizon")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote %s" % out_path)

    # Match the official metric exactly. cloth_eval.evaluate / cfd_eval.evaluate
    # define `mse_<N>_steps` as the MEAN over steps 1..N, not the value at step N:
    #     error = reduce_mean((pred - gt)**2, axis=-1)
    #     mse_N = reduce_mean(error[1:N+1])
    # The plotted curve above is the per-step error (the paper's figure); this
    # table is the aggregate the official eval prints, so the two are comparable.
    print("\nmean per-step MSE at the reported horizons "
          "(official cloth_eval metric):")
    for horizon in (1, 10, 20, 50, 100, 200):
        if horizon <= len(mean_curve):
            print("  mse_%d_steps = %.6e" % (horizon, np.nanmean(mean_curve[1:horizon + 1])))
    print("\nper-step MSE at exactly step N (for reference):")
    for horizon in (1, 10, 20, 50, 100, 200):
        if horizon <= len(mean_curve):
            print("  step %-4d    = %.6e" % (horizon, mean_curve[horizon - 1]))


def _trajectory_arrays(traj):
    """Return render arrays and static mesh connectivity for one trajectory."""
    is_cloth = _is_cloth(traj)
    pred = traj["pred_pos"] if is_cloth else traj["pred_velocity"]
    gt = traj["gt_pos"] if is_cloth else traj["gt_velocity"]
    faces = traj["faces"]
    if faces.ndim == 3:
        faces = faces[0]
    return is_cloth, pred, gt, faces


def _frame_steps(num_steps, skip, max_frames):
    """Sample the whole rollout rather than silently truncating its tail."""
    candidates = np.arange(0, num_steps, max(1, skip), dtype=int)
    if len(candidates) <= max_frames:
        return candidates.tolist()
    indices = np.linspace(0, len(candidates) - 1, max_frames).round().astype(int)
    return candidates[indices].tolist()


def _bounds(pred, gt):
    lo = np.minimum(pred.min(axis=(0, 1)), gt.min(axis=(0, 1))).astype(float)
    hi = np.maximum(pred.max(axis=(0, 1)), gt.max(axis=(0, 1))).astype(float)
    span = np.maximum(hi - lo, 1e-6)
    return lo - 0.04 * span, hi + 0.04 * span


def _style_3d_axis(ax, lo, hi, title):
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.maximum(hi - lo, 1e-6))
    ax.view_init(elev=24, azim=-58)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=2)
    ax.set_axis_off()


def render_gif(trajectories, out_path, skip, max_frames, fps=10,
               trajectory_index=0):
    """Enhanced prediction / ground truth / spatial-error comparison."""
    from matplotlib import animation, colors, cm

    traj = trajectories[trajectory_index]
    is_cloth, pred, gt, faces = _trajectory_arrays(traj)
    steps = _frame_steps(len(pred), skip, max_frames)
    lo, hi = _bounds(pred, gt)
    vertex_error = np.linalg.norm(pred - gt, axis=-1)
    mse = np.mean((pred - gt) ** 2, axis=tuple(range(1, pred.ndim)))
    cumulative_mse = np.cumsum(mse) / (np.arange(len(mse)) + 1)
    error_ceiling = max(float(np.percentile(vertex_error, 99)), 1e-12)
    norm = colors.Normalize(vmin=0.0, vmax=error_ceiling)

    if not is_cloth:
        # Retain a useful fallback for CFD-like 2-D data.
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor="#f7f7f7")
    else:
        fig = plt.figure(figsize=(13, 4.6), facecolor="#f7f7f7")
        axes = [fig.add_subplot(131, projection="3d"),
                fig.add_subplot(132, projection="3d"),
                fig.add_subplot(133, projection="3d")]
    fig.subplots_adjust(left=0.01, right=0.94, bottom=0.02, top=0.86, wspace=0.02)
    colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="magma"),
                           ax=axes, fraction=0.018, pad=0.015)
    colorbar.set_label("vertex position error", fontsize=9)

    def animate(i):
        step = steps[i]
        for ax in axes:
            ax.cla()
        if is_cloth:
            axes[0].plot_trisurf(*pred[step].T, triangles=faces,
                                 color="#2b8cbe", linewidth=0, antialiased=False,
                                 shade=True)
            axes[1].plot_trisurf(*gt[step].T, triangles=faces,
                                 color="#31a354", linewidth=0, antialiased=False,
                                 shade=True)
            surface = axes[2].plot_trisurf(*pred[step].T, triangles=faces,
                                           cmap="magma", norm=norm, linewidth=0,
                                           antialiased=False, shade=False)
            surface.set_array(vertex_error[step][faces].mean(axis=1))
            for ax, title in zip(axes,
                                 ("PREDICTION", "GROUND TRUTH", "ERROR HEATMAP")):
                _style_3d_axis(ax, lo, hi, title)
        else:
            for ax, data, title in zip(axes[:2], (pred, gt),
                                       ("PREDICTION", "GROUND TRUTH")):
                ax.tripcolor(data[step, :, 0], data[step, :, 1], faces,
                             data[step, :, 0], shading="gouraud", cmap="viridis")
                ax.set_title(title, fontweight="bold")
            axes[2].tripcolor(pred[step, :, 0], pred[step, :, 1], faces,
                              vertex_error[step], shading="gouraud", cmap="magma",
                              norm=norm)
            axes[2].set_title("ERROR HEATMAP", fontweight="bold")
            for ax in axes:
                ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
                ax.set_aspect("equal"); ax.set_axis_off()
        fig.suptitle("Trajectory %02d  |  step %03d / %03d  |  MSE %.3e  |  cumulative %.3e" %
                     (trajectory_index, step, len(pred) - 1, mse[step], cumulative_mse[step]),
                     fontsize=12, fontweight="bold", y=0.96)
        return axes

    anim = animation.FuncAnimation(fig, animate, frames=len(steps), interval=1000 / fps)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=85)
    plt.close(fig)
    print("wrote %s (%d frames, trajectory %d)" %
          (out_path, len(steps), trajectory_index))


def render_overlay_gif(traj, out_path, skip, max_frames, fps=10,
                       trajectory_index=0):
    """Overlay prediction and truth in contrasting translucent colors."""
    from matplotlib import animation
    from matplotlib.patches import Patch

    is_cloth, pred, gt, faces = _trajectory_arrays(traj)
    steps = _frame_steps(len(pred), skip, max_frames)
    lo, hi = _bounds(pred, gt)
    mse = np.mean((pred - gt) ** 2, axis=tuple(range(1, pred.ndim)))

    if is_cloth:
        fig = plt.figure(figsize=(7.2, 6.2), facecolor="#10151c")
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor="#10151c")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.88)
    fig.legend(handles=[Patch(color="#20a4f3", label="Prediction"),
                        Patch(color="#ff9f1c", label="Ground truth")],
               loc="lower center", ncol=2, frameon=False, labelcolor="white")

    def animate(i):
        step = steps[i]
        ax.cla()
        if is_cloth:
            ax.plot_trisurf(*gt[step].T, triangles=faces, color="#ff9f1c",
                            alpha=0.58, linewidth=0, antialiased=False, shade=True)
            ax.plot_trisurf(*pred[step].T, triangles=faces, color="#20a4f3",
                            alpha=0.58, linewidth=0, antialiased=False, shade=True)
            _style_3d_axis(ax, lo, hi, "PREDICTION / GROUND TRUTH OVERLAY")
            ax.set_facecolor("#10151c")
        else:
            ax.triplot(gt[step, :, 0], gt[step, :, 1], faces,
                       color="#ff9f1c", alpha=.55, linewidth=.35)
            ax.triplot(pred[step, :, 0], pred[step, :, 1], faces,
                       color="#20a4f3", alpha=.55, linewidth=.35)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_aspect("equal"); ax.set_axis_off()
        fig.suptitle("Trajectory %02d  |  step %03d / %03d  |  MSE %.3e" %
                     (trajectory_index, step, len(pred) - 1, mse[step]),
                     color="white", fontsize=13, fontweight="bold", y=0.96)
        return ax,

    anim = animation.FuncAnimation(fig, animate, frames=len(steps), interval=1000 / fps)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=85,
              savefig_kwargs={"facecolor": "#10151c"})
    plt.close(fig)
    print("wrote %s (%d frames, trajectory %d)" %
          (out_path, len(steps), trajectory_index))


def render_batch(trajectories, out_dir, skip, max_frames, fps):
    """Generate both diagnostic views for every available trajectory."""
    os.makedirs(out_dir, exist_ok=True)
    for index, traj in enumerate(trajectories):
        stem = "traj_%02d" % index
        render_gif(trajectories, os.path.join(out_dir, stem + "_compare.gif"),
                   skip, max_frames, fps=fps, trajectory_index=index)
        render_overlay_gif(traj, os.path.join(out_dir, stem + "_overlay.gif"),
                           skip, max_frames, fps=fps, trajectory_index=index)


def main(unused_argv):
    trajectories = _load(FLAGS.rollout_path)
    print("loaded %d trajectory(ies) from %s" % (len(trajectories), FLAGS.rollout_path))
    first = trajectories[0]
    for key, val in first.items():
        print("  %-12s %s" % (key, getattr(val, "shape", type(val))))

    if FLAGS.mse_out:
        plot_mse(trajectories, FLAGS.mse_out)
    if FLAGS.gif_out:
        render_gif(trajectories, FLAGS.gif_out, FLAGS.skip, FLAGS.max_gif_frames,
                   fps=FLAGS.gif_fps)
    if FLAGS.batch_gif_dir:
        render_batch(trajectories, FLAGS.batch_gif_dir, FLAGS.skip,
                     FLAGS.max_gif_frames, FLAGS.gif_fps)


if __name__ == "__main__":
    app.run(main)
