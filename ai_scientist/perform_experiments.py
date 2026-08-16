import json
import os
import os.path as osp
import shutil
import subprocess
import sys
from subprocess import TimeoutExpired


MAX_ITERS = 4
MAX_RUNS = 5
MAX_STDERR_OUTPUT = 1500


coder_prompt = """Your goal is to implement the following idea: {title}.
The proposed experiment is as follows: {idea}.
You are given a total of up to {max_runs} runs to complete the necessary experiments. You do not need to use all {max_runs}.

First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run.

Note that we already provide the vanilla baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:

{baseline_results}

After you complete each change, we will run the command `python experiment.py --out_dir=run_i` where i is the run number and evaluate the results.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.

Implement the proposed experiment by editing the existing `experiment.py`.

Modify the existing experiment in place.
Do not append a second complete implementation of the experiment.
Do not duplicate the full file, imports, main function, or command-line entry point.

The output directory name (`run_i`) is only an orchestration identifier.

Do not make experimental behavior depend on the output directory or run number.
Do not branch on values such as `run_1`, `run_2`, `run_3`, etc.

Each new experimental configuration must be implemented by directly editing
`experiment.py` between runs. The current `experiment.py` must define the
scientific configuration that is executed when we call:

`python experiment.py --out_dir=run_i`

The `--out_dir` argument must control only where outputs are saved, not which
scientific method, ablation, hyperparameter, or experimental condition is run.

Every successful run must create:

`run_i/final_info.json`

The primary metrics in `final_info.json` must use the existing structure:

{{
    "metric_name": {{
        "means": <numeric value>,
        "stds": <numeric value>
    }}
}}

You may create additional experiment-specific output files when needed, but they must not replace `final_info.json`.

You can then implement the next thing on your list.
"""


# ================================================================
# RUN EXPERIMENT
# ================================================================

def run_experiment(
    folder_name,
    run_num,
    timeout=7200,
):

    cwd = osp.abspath(
        folder_name
    )

    # ------------------------------------------------------------
    # COPY THE EXACT CODE USED FOR THIS RUN
    # ------------------------------------------------------------

    shutil.copy(
        osp.join(
            folder_name,
            "experiment.py",
        ),
        osp.join(
            folder_name,
            f"run_{run_num}.py",
        ),
    )

    # ------------------------------------------------------------
    # LAUNCH EXPERIMENT
    # ------------------------------------------------------------

    command = [
        "python",
        "experiment.py",
        f"--out_dir=run_{run_num}",
    ]

    try:

        result = subprocess.run(
            command,
            cwd=cwd,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )

        if result.stderr:
            print(
                result.stderr,
                file=sys.stderr,
            )

        # --------------------------------------------------------
        # FAILED RUN
        # --------------------------------------------------------

        if result.returncode != 0:

            print(
                f"Run {run_num} failed with return code "
                f"{result.returncode}"
            )

            run_dir = osp.join(
                cwd,
                f"run_{run_num}",
            )

            if osp.exists(
                run_dir
            ):
                shutil.rmtree(
                    run_dir
                )

            stderr_output = (
                result.stderr
            )

            if (
                len(stderr_output)
                > MAX_STDERR_OUTPUT
            ):
                stderr_output = (
                    "..."
                    + stderr_output[
                        -MAX_STDERR_OUTPUT:
                    ]
                )

            next_prompt = (
                "Run failed with the following error:\n\n"
                f"{stderr_output}\n\n"
                "Fix the error in `experiment.py` while preserving "
                "the intended experiment."
            )

            return (
                result.returncode,
                next_prompt,
            )

        # --------------------------------------------------------
        # SUCCESSFUL RUN
        # --------------------------------------------------------

        final_info_path = osp.join(
            cwd,
            f"run_{run_num}",
            "final_info.json",
        )

        # --------------------------------------------------------
        # FINAL_INFO GUARD
        # --------------------------------------------------------

        if not osp.exists(
            final_info_path
        ):

            print(
                f"Run {run_num} completed but did not produce "
                "the required final_info.json."
            )

            next_prompt = f"""
Run {run_num} executed successfully, but the required file

`run_{run_num}/final_info.json`

was not created.

This violates the experiment output contract.

Fix `experiment.py` so that every successful run creates
`final_info.json`.

Preserve the experimental logic that was just implemented.

Do not redesign or replace the experiment.

You may keep any supplementary experiment-specific JSON files.

`final_info.json` must contain the primary evaluation metrics using:

{{
    "metric_name": {{
        "means": <numeric value>,
        "stds": <numeric value>
    }}
}}

Fix only the missing or incompatible final output generation.
"""

            return (
                1,
                next_prompt,
            )

        # --------------------------------------------------------
        # READ ALL PRIMARY METRICS
        # --------------------------------------------------------

        with open(
            final_info_path,
            "r",
            encoding="utf-8",
        ) as f:
            results = json.load(
                f
            )

        # Extract means for feedback to the scientist.
        # Keep every metric present in final_info.json.
        result_means = {}

        for key, value in results.items():

            if (
                isinstance(
                    value,
                    dict,
                )
                and "means" in value
            ):
                result_means[
                    key
                ] = value[
                    "means"
                ]

        next_prompt = f"""Run {run_num} completed. Here are the results:

{result_means}

Decide if you need to re-plan your experiments given the result
(you often will not need to).

Someone else will be using `notes.txt` to perform a writeup on this
in the future. Please include all relevant information for the writeup
on Run {run_num}, including the experiment description, results, and
important scientific interpretation.

Then implement the next thing on your list.

Implement the next experimental configuration directly in `experiment.py`.

Do not make the scientific behavior depend on
`run_{run_num + 1}` or any other output-directory name.
The output directory is only used to save the results.

We will then run the command
`python experiment.py --out_dir=run_{run_num + 1}`.

YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT,
DO NOT ADD ADDITIONAL COMMAND LINE ARGS.

If you are finished with experiments, respond with 'ALL_COMPLETED'."""

        return (
            result.returncode,
            next_prompt,
        )

    except TimeoutExpired:

        print(
            f"Run {run_num} timed out "
            f"after {timeout} seconds"
        )

        run_dir = osp.join(
            cwd,
            f"run_{run_num}",
        )

        if osp.exists(
            run_dir
        ):
            shutil.rmtree(
                run_dir
            )

        next_prompt = f"""
Run {run_num} timed out after {timeout} seconds.

Modify `experiment.py` so the intended experiment can complete within
the available runtime.

Preserve the scientific purpose of the experiment.
"""

        return (
            1,
            next_prompt,
        )


# ================================================================
# RUN PLOTTING
# ================================================================

def run_plotting(
    folder_name,
    timeout=600,
):

    cwd = osp.abspath(
        folder_name
    )

    command = [
        "python",
        "plot.py",
    ]

    try:

        result = subprocess.run(
            command,
            cwd=cwd,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )

        if result.stderr:
            print(
                result.stderr,
                file=sys.stderr,
            )

        if result.returncode != 0:

            print(
                "Plotting failed with return code "
                f"{result.returncode}"
            )

            next_prompt = (
                "Plotting failed with the following error:\n\n"
                f"{result.stderr}"
            )

        else:

            next_prompt = ""

        return (
            result.returncode,
            next_prompt,
        )

    except TimeoutExpired:

        print(
            "Plotting timed out after "
            f"{timeout} seconds"
        )

        next_prompt = (
            "Plotting timed out after "
            f"{timeout} seconds"
        )

        return (
            1,
            next_prompt,
        )


# ================================================================
# PERFORM EXPERIMENTS
# ================================================================

def perform_experiments(
    idea,
    folder_name,
    coder,
    baseline_results,
) -> bool:

    # ============================================================
    # EXPERIMENT LOOP
    # ============================================================

    current_iter = 0
    run = 1

    next_prompt = coder_prompt.format(
        title=idea[
            "Title"
        ],
        idea=idea[
            "Experiment"
        ],
        max_runs=MAX_RUNS,
        baseline_results=baseline_results,
    )

    while (
        run
        < MAX_RUNS + 1
    ):

        if (
            current_iter
            >= MAX_ITERS
        ):

            print(
                "Max iterations reached"
            )

            break

        # --------------------------------------------------------
        # SAVE EXPERIMENT.PY BEFORE AIDER EDIT
        # --------------------------------------------------------

        experiment_path = osp.join(
            folder_name,
            "experiment.py",
        )

        with open(
            experiment_path,
            "r",
            encoding="utf-8",
        ) as f:
            before_code = f.read()

        # --------------------------------------------------------
        # ASK SCIENTIST TO IMPLEMENT NEXT CHANGE
        # --------------------------------------------------------

        coder_out = coder.run(
            next_prompt
        )

        print(
            coder_out
        )

        if (
            "ALL_COMPLETED"
            in coder_out
        ):

            break

        # --------------------------------------------------------
        # EXPERIMENT CHANGE GUARD
        # --------------------------------------------------------

        with open(
            experiment_path,
            "r",
            encoding="utf-8",
        ) as f:
            after_code = f.read()

        if (
            before_code
            == after_code
        ):

            print(
                "experiment.py was not modified."
            )

            next_prompt = (
                "You did not modify experiment.py. "
                "You must implement the proposed experiment by editing "
                "experiment.py now. "
                "Do not only describe your plan and do not ask for confirmation."
            )

            current_iter += 1

            continue

        # --------------------------------------------------------
        # RUN IMPLEMENTED EXPERIMENT
        # --------------------------------------------------------

        (
            return_code,
            next_prompt,
        ) = run_experiment(
            folder_name,
            run,
        )

        if (
            return_code
            == 0
        ):

            run += 1
            current_iter = 0

        current_iter += 1

    # ------------------------------------------------------------
    # FAILED TO COMPLETE EXPERIMENT LOOP
    # ------------------------------------------------------------

    if (
        current_iter
        >= MAX_ITERS
    ):

        print(
            "Not all experiments completed."
        )

        return False

    # ============================================================
    # PLOTTING
    # ============================================================

    current_iter = 0

    plot_path = osp.join(
        folder_name,
        "plot.py",
    )

    # Save the original plotting code only for logging purposes.
    with open(
        plot_path,
        "r",
        encoding="utf-8",
    ) as f:
        before_plot_code = f.read()

    next_prompt = f"""
Great job! Please modify `plot.py` to generate the most relevant plots
for the final writeup of this experiment.

Research title:

{idea["Title"]}

Research idea:

{idea["Experiment"]}

Review the completed experimental runs, `notes.txt`, and the outputs
produced by the experiment.

Make sure `plot.py` reflects the experiment that was actually performed.
If the experiment introduced new results or comparisons that the existing
plots do not represent, update the plotting code accordingly.

When labeling experimental runs, derive each label from the actual
experiment history, `notes.txt`, and the code or configuration used for
that run.

Do not infer a run's scientific meaning from its run number alone.

Do not hardcode assumptions such as:
- run_1 = one method,
- run_2 = another method,
- run_3 = a specific hyperparameter value,

unless that exact mapping is supported by the completed experiment history.

If the exact scientific configuration of a run cannot be verified, use a
neutral label such as "Experimental configuration 1" rather than inventing
a scientific description.

Only include completed runs that are scientifically relevant to the final
figures.

Preserve useful existing plots.

Only use results that actually exist.
Do not fabricate missing results.

We will run:

`python plot.py`
"""

    while True:

        # --------------------------------------------------------
        # LET AIDER ADAPT/FIX PLOTTING
        # --------------------------------------------------------

        coder_out = coder.run(
            next_prompt
        )

        print(
            coder_out
        )

        # --------------------------------------------------------
        # LOG WHETHER PLOT.PY CHANGED
        # --------------------------------------------------------

        with open(
            plot_path,
            "r",
            encoding="utf-8",
        ) as f:
            after_plot_code = f.read()

        if (
            before_plot_code
            == after_plot_code
        ):

            print(
                "plot.py was not modified."
            )

        else:

            print(
                "plot.py was updated for the completed experiment."
            )

        # --------------------------------------------------------
        # RUN PLOTTING
        # --------------------------------------------------------

        (
            return_code,
            plotting_error,
        ) = run_plotting(
            folder_name
        )

        current_iter += 1

        if (
            return_code
            == 0
        ):

            break

        if (
            current_iter
            >= MAX_ITERS
        ):

            break

        # --------------------------------------------------------
        # PLOTTING ERROR REPAIR
        # -------------------------------------------------------
        next_prompt = f"""
`plot.py` failed with the following error:

{plotting_error}

Fix the plotting error in `plot.py`.

Preserve the scientifically relevant plots that have already been
implemented.

Do not remove experiment-specific plots simply to make the script run.

Preserve scientifically accurate run labels. Do not invent a run's
configuration from its run number.

Do not fabricate missing results.

We will run:

`python plot.py`
"""

        # Compare subsequent edits against the latest plot.py.
        with open(
            plot_path,
            "r",
            encoding="utf-8",
        ) as f:
            before_plot_code = f.read()

    if (
        current_iter
        >= MAX_ITERS
        and return_code != 0
    ):

        print(
            "Plotting did not complete successfully."
        )

    # ============================================================
    # NOTES
    # ============================================================

    generated_figures = sorted(
        filename
        for filename in os.listdir(folder_name)
        if filename.lower().endswith(".png")
    )

    next_prompt = f"""
Please modify `notes.txt` with an in-depth description of each generated
figure and its exact filename.

The following figure files were actually generated:

{generated_figures}

Describe only figures from this list.

For every figure, include:

- the exact filename,
- what the figure shows,
- what experimental setting, method, condition, or comparison it represents,
- what the important visual elements mean,
- the main scientific purpose or takeaway,
- and the manuscript section where the figure would be most appropriate.

Determine the appropriate manuscript section from the scientific role
of the figure.

Use your scientific judgment rather than relying on a fixed list of
plot types or experiment-specific assumptions.

Do not invent figure filenames, results, interpretations, or experimental
details.

Do not treat internal run names, directory names, or pipeline artifacts
as scientific concepts. When possible, describe the actual scientific
configuration represented by the figure.

Clearly state the recommended manuscript section for every figure.

Use a consistent format such as:

Figure: <exact filename>
Recommended section: <manuscript section>
Description: <what the figure shows>
Scientific takeaway: <why the figure matters>

Somebody else will use `notes.txt` to write the scientific report and place
the figures in the appropriate manuscript sections.
"""
    coder.run(
        next_prompt
    )

    return True