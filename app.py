import glob
import os
import shutil
import subprocess
import tempfile

import gradio as gr

CWD = os.path.dirname(os.path.abspath(__file__))


def get_examples() -> list[list[str]]:
    example_dir = os.path.join(CWD, "examples")
    if not os.path.exists(example_dir):
        return []
    extensions = (".obj",)
    return [
        [os.path.join(example_dir, f)] for f in os.listdir(example_dir) if any(f.endswith(ext) for ext in extensions)
    ]


def run_cmd(cmd: list[str], cwd: str = CWD) -> tuple[bool, str]:
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logs = []
    if process.stdout:
        for line in process.stdout:
            print(line, end="")
            logs.append(line)
    process.wait()

    return process.returncode == 0, "".join(logs)


def generate_skeleton(input_path: str, progress=gr.Progress()) -> tuple[list[str], str]:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    input_name = os.path.basename(input_path)
    input_name_no_ext = os.path.splitext(input_name)[0]
    temp_dir = tempfile.mkdtemp(prefix="puppeteer_pipeline_")
    input_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "results")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    temp_input_path = os.path.join(input_dir, input_name)
    shutil.copy(input_path, temp_input_path)

    progress(0.5, desc="Stage 1/3: Generating Skeleton...")
    success, log = run_cmd(
        [
            "python",
            "demo.py",
            "--input_dir",
            input_dir,
            "--pretrained_weights",
            "skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth",
            "--output_dir",
            output_dir,
            "--save_name",
            "skel_results",
            "--input_pc_num",
            "8192",
            "--save_render",
            "--apply_marching_cubes",
            "--joint_token",
            "--seq_shuffle",
        ],
        os.path.join(CWD, "skeleton"),
    )
    if success:
        gr.Info("Skeleton generation completed.")
    else:
        raise gr.Error(f"Skeleton generation failed:\n{log}")

    render_dir = os.path.join(output_dir, "skel_results", "render_results", f"{input_name_no_ext}_pred")
    skel_images = sorted(glob.glob(os.path.join(render_dir, "*.png")))

    return skel_images, temp_dir


def generate_skinning(input_path: str, temp_dir: str, progress=gr.Progress()) -> str:
    input_name = os.path.basename(input_path)
    input_name_no_ext = os.path.splitext(input_name)[0]
    input_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "results")
    skel_results_dir = os.path.join(output_dir, "skel_results")
    skeletons_dir = os.path.join(output_dir, "skeletons")
    os.makedirs(skeletons_dir, exist_ok=True)
    pred_txt = os.path.join(skel_results_dir, f"{input_name_no_ext}_pred.txt")
    shutil.copy(pred_txt, os.path.join(skeletons_dir, f"{input_name_no_ext}.txt"))

    progress(0.5, desc="Stage 2/3: Generating Skinning...")
    success, log = run_cmd(
        [
            "torchrun",
            "--nproc_per_node=1",
            "--master_port=10009",
            "main.py",
            "--num_workers",
            "1",
            "--batch_size",
            "1",
            "--generate",
            "--save_skin_npy",
            "--pretrained_weights",
            "skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth",
            "--input_skel_folder",
            skeletons_dir,
            "--mesh_folder",
            input_dir,
            "--post_filter",
            "--depth",
            "1",
            "--save_folder",
            os.path.join(output_dir, "skin_results"),
        ],
        os.path.join(CWD, "skinning"),
    )
    if success:
        gr.Info("Skinning generation completed.")
    else:
        raise gr.Error(f"Skinning generation failed:\n{log}")

    skin_results_dir = os.path.join(output_dir, "skin_results", "generate")
    skin_txt_path = os.path.join(skin_results_dir, f"{input_name_no_ext}_skin.txt")
    final_rigging_dir = os.path.join(output_dir, "final_rigging")
    os.makedirs(final_rigging_dir, exist_ok=True)
    rigging_txt_path = os.path.join(final_rigging_dir, f"{input_name_no_ext}.txt")
    shutil.copy(skin_txt_path, rigging_txt_path)

    return rigging_txt_path


def export(input_path: str, temp_dir: str, progress=gr.Progress()) -> str:
    input_name = os.path.basename(input_path)
    input_name_no_ext = os.path.splitext(input_name)[0]
    input_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "results")
    temp_input_path = os.path.join(input_dir, input_name)
    final_rigging_dir = os.path.join(output_dir, "final_rigging")
    rigging_txt_path = os.path.join(final_rigging_dir, f"{input_name_no_ext}.txt")
    rigged_output_path = os.path.join(final_rigging_dir, f"{input_name_no_ext}.fbx")

    progress(0.5, desc="Stage 3/3: Exporting...")
    success, log = run_cmd(
        ["python", "export.py", "--mesh", temp_input_path, "--rig", rigging_txt_path, "--output", rigged_output_path],
        CWD,
    )
    if success:
        gr.Success("Pipeline finished successfully!")
    else:
        raise gr.Error(f"Exporting failed:\n{log}")

    return rigged_output_path


if __name__ == "__main__":
    subprocess.run(["python", "skeleton/download.py"], cwd=CWD, check=True)
    subprocess.run(["python", "skinning/download.py"], cwd=CWD, check=True)
    subprocess.run(["python", "animation/download.py"], cwd=CWD, check=True)

    with gr.Blocks(title="Puppeteer") as app:
        gr.Markdown("# Puppeteer: Rig and Animate Your 3D Models")
        gr.Markdown("[Code](https://github.com/Seed3D/Puppeteer)")

        temp_dir = gr.State()

        with gr.Row():
            with gr.Column():
                input_3d = gr.Model3D(label="Input Mesh")
                run_btn = gr.Button("Run", variant="primary")

                gr.Examples(examples=get_examples(), inputs=input_3d, label="Examples")

            with gr.Column():
                output_skel_imgs = gr.Gallery(label="Skeleton Render Results", columns=2, object_fit="contain")
                output_txt = gr.File(label="Rigging Results")
                output_3d = gr.File(label="Rigged Mesh")

        run_btn.click(
            fn=lambda: (None, None, None),
            outputs=[output_skel_imgs, output_txt, output_3d],
        ).success(
            fn=generate_skeleton,
            inputs=input_3d,
            outputs=[output_skel_imgs, temp_dir],
        ).success(
            fn=generate_skinning,
            inputs=[input_3d, temp_dir],
            outputs=output_txt,
        ).success(
            fn=export,
            inputs=[input_3d, temp_dir],
            outputs=output_3d,
        )

    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True, share=False)
