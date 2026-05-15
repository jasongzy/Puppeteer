import os

from huggingface_hub import hf_hub_download

workdir = os.path.dirname(os.path.abspath(__file__))

file_path = hf_hub_download(
    repo_id="Maikou/Michelangelo",
    filename="checkpoints/aligned_shape_latents/shapevae-256.ckpt",
    local_dir=os.path.join(workdir, "third_partys/Michelangelo"),
)

file_path = hf_hub_download(
    repo_id="Seed3D/Puppeteer",
    filename="skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth",
    local_dir=workdir,
)

file_path = hf_hub_download(
    repo_id="Seed3D/Puppeteer",
    filename="skeleton_ckpts/puppeteer_skeleton_wo_diverse_pose.pth",
    local_dir=workdir,
)

file_path = hf_hub_download(
    repo_id="Seed3D/Puppeteer",
    filename="skeleton_ckpts/puppeteer_skeleton_w_diverse_pose_bone_token.pth",
    local_dir=workdir,
)
