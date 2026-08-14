import argparse
import json
import os
import os.path as osp
import re
import shutil
import subprocess
from typing import Optional, Tuple

from ai_scientist.generate_ideas import search_for_papers
from ai_scientist.llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS

# GENERATE LATEX
def generate_latex(coder, folder_name, pdf_file, timeout=300, num_error_corrections=5):
    folder = osp.abspath(folder_name)
    cwd = osp.join(folder, "latex")  # Fixed potential issue with path
    writeup_file = osp.join(cwd, "template.tex")

    # Check all references are valid and in the references.bib file
    with open(writeup_file, "r") as f:
        tex_text = f.read()
    cites = re.findall(r"\\cite[a-z]*{([^}]*)}", tex_text)
    references_bib = re.search(
        r"\\begin{filecontents}{references.bib}(.*?)\\end{filecontents}",
        tex_text,
        re.DOTALL,
    )
    if references_bib is None:
        print("No references.bib found in template.tex")
        return
    bib_text = references_bib.group(1)
    cites = [cite.strip() for item in cites for cite in item.split(",")]
    for cite in cites:
        if cite not in bib_text:
            print(f"Reference {cite} not found in references.")
            prompt = f"""Reference {cite} not found in references.bib. Is this included under a different name?
If so, please modify the citation in template.tex to match the name in references.bib at the top. Otherwise, remove the cite."""
            coder.run(prompt)

    # Check all included figures are actually in the directory.
    with open(writeup_file, "r") as f:
        tex_text = f.read()
    referenced_figs = re.findall(r"\\includegraphics.*?{(.*?)}", tex_text)
    all_figs = [f for f in os.listdir(folder) if f.endswith(".png")]
    for figure in referenced_figs:
        if figure not in all_figs:
            print(f"Figure {figure} not found in directory.")
            prompt = f"""The image {figure} not found in the directory. The images in the directory are: {all_figs}.
Please ensure that the figure is in the directory and that the filename is correct. Check the notes to see what each figure contains."""
            coder.run(prompt)

    # Remove duplicate figures.
    with open(writeup_file, "r") as f:
        tex_text = f.read()
    referenced_figs = re.findall(r"\\includegraphics.*?{(.*?)}", tex_text)
    duplicates = {x for x in referenced_figs if referenced_figs.count(x) > 1}
    if duplicates:
        for dup in duplicates:
            print(f"Duplicate figure found: {dup}.")
            prompt = f"""Duplicate figures found: {dup}. Ensure any figure is only included once.
If duplicated, identify the best location for the figure and remove any other."""
            coder.run(prompt)

    # Remove duplicate section headers.
    with open(writeup_file, "r") as f:
        tex_text = f.read()
    sections = re.findall(r"\\section{([^}]*)}", tex_text)
    duplicates = {x for x in sections if sections.count(x) > 1}
    if duplicates:
        for dup in duplicates:
            print(f"Duplicate section header found: {dup}")
            prompt = f"""Duplicate section header found: {dup}. Ensure any section header is declared once.
If duplicated, identify the best location for the section header and remove any other."""
            coder.run(prompt)

    # Iteratively fix any LaTeX bugs
    for i in range(num_error_corrections):
        check_output = os.popen(
            f"chktex {writeup_file} -q -n2 -n24 -n13 -n1"
        ).read()

        if not check_output:
            break

        # Save file before asking Aider to fix it.
        with open(writeup_file, "r") as f:
            before_tex = f.read()

        prompt = f"""Please fix the following LaTeX errors in `template.tex`
    guided by the output of `chktex`:

    {check_output}

    Make the minimal fix required.
    Do not remove or change any packages.
    Pay attention to accidental HTML syntax such as </end instead of \\end.
    """

        coder.run(prompt)

        # Check whether Aider actually changed the file.
        with open(writeup_file, "r") as f:
            after_tex = f.read()

        if before_tex == after_tex:
            print(
                "Aider did not modify template.tex. "
                "Stopping LaTeX correction loop."
            )
            break
    compile_latex(cwd, pdf_file, timeout=timeout)



def compile_latex(cwd, pdf_file, timeout=30):
    print("GENERATING LATEX")

    commands = [
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["bibtex", "template"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
    ]

    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            print("Standard Output:\n", result.stdout)
            print("Standard Error:\n", result.stderr)
        except subprocess.TimeoutExpired:
            print(f"Latex timed out after {timeout} seconds")
        except subprocess.CalledProcessError as e:
            print(f"Error running command {' '.join(command)}: {e}")

    print("FINISHED GENERATING LATEX")

    # Attempt to move the PDF to the desired location
    try:
        shutil.move(osp.join(cwd, "template.pdf"), pdf_file)
    except FileNotFoundError:
        print("Failed to rename PDF.")


per_section_tips = {
    "Abstract": """
- TL;DR of the paper
- What are we trying to do and why is it relevant?
- Why is this hard? 
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)

Please make sure the abstract reads smoothly and is well-motivated. This should be one continuous paragraph with no breaks between the lines.
""",
    "Introduction": """
- Longer version of the Abstract, i.e. of the entire paper
- What are we trying to do and why is it relevant?
- Why is this hard? 
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)
- New trend: specifically list your contributions as bullet points
- Extra space? Future work!
""",
    "Related Work": """
- Academic siblings of our work, i.e. alternative attempts in literature at trying to solve the same problem. 
- Goal is to “Compare and contrast” - how does their approach differ in either assumptions or method? If their method is applicable to our Problem Setting I expect a comparison in the experimental section. If not, there needs to be a clear statement why a given method is not applicable. 
- Note: Just describing what another paper is doing is not enough. We need to compare and contrast.
""",
    "Background": """
- Academic Ancestors of our work, i.e. all concepts and prior work that are required for understanding our method. 
- Usually includes a subsection, Problem Setting, which formally introduces the problem setting and notation (Formalism) for our method. Highlights any specific assumptions that are made that are unusual. 
- Note: If our paper introduces a novel problem setting as part of its contributions, it's best to have a separate Section.
""",
    "Method": """
- What we do and why we do it, using the formalism introduced in the Problem Setting.
- Describe only methods actually implemented in the experiment code.
- If an implementation detail cannot be verified, omit it rather than guessing.
""",
    "Experimental Setup": """
- How do we test that our stuff works? Introduces a specific instantiation of the Problem Setting and specific implementation details of our Method for this Problem Setting.
- Do not imagine unknown hardware details.
- Includes a description of the dataset, evaluation metrics, important hyperparameters, and implementation details.
""",
  "Results": """
- Present the completed experimental findings clearly and concisely.
- Use only results that were actually produced by the completed experiments.
- Report the scientifically relevant metrics and observations supported by
  the experiment artifacts.
- Compare methods, configurations, conditions, or baselines only when the
  recorded outputs support the comparison.
- Do not automatically choose or describe a best configuration based on a
  single metric.
- Do not invent missing metrics, experiments, uncertainty estimates,
  statistical tests, or significance claims.
- Include scientifically relevant figures assigned to this section.
- Discuss negative, mixed, or inconclusive findings when they are important.
- Discuss limitations supported by the experiments.
- Do not mention internal pipeline artifacts such as run directories,
  notes.txt, JSON filenames, or implementation bookkeeping in the
  scientific prose.
""",

    "Conclusion": """
- Brief recap of the entire paper.
- To keep going with the analogy, you can think of future work as (potential) academic offspring.
""",
}

error_list = """- Unenclosed math symbols

- Only reference figures that exist in our directory
- LaTeX syntax errors
- Numerical results that do not come from explicit experiments and logs
- Claims of improvement, robustness, or statistical significance not supported by experiment artifacts
- Confusing comparisons between experimental runs with comparisons against baselines within a run
- Claims about baseline performance for metrics that are not explicitly reported for that baseline
- Claims that different runs represent random repetitions unless this is explicitly supported by the experiment artifacts
- References to internal pipeline artifacts such as `notes.txt` in the scientific manuscript
- Repeatedly defined figure labels
- References to papers that are not in the .bib file, DO NOT ADD ANY NEW CITATIONS!
- Unnecessary verbosity or repetition, unclear text
- Results or insights in the `notes.txt` that have not yet need included
- Any relevant figures that have not yet been included in the text
- Closing any \\begin{{figure}} with a \\end{{figure}} and \\begin{{table}} with a \\end{{table}}, etc.
- Duplicate headers, e.g. duplicated \\section{{Introduction}} or \\end{{document}}
- Unescaped symbols, e.g. shakespeare_char should be shakespeare\\_char in text
- Incorrect closing of environments, e.g. </end{{figure}}> instead of \\end{{figure}}
"""

refinement_prompt = (
    """Great job! Now criticize and refine only the {section} that you just wrote.
Make this complete in this pass, do not leave any placeholders.

Pay particular attention to fixing any errors such as:
"""
    + error_list
)

second_refinement_prompt = (
    """Criticize and refine the {section} only. Recall the advice:
{tips}
Make this complete in this pass, do not leave any placeholders.

Pay attention to how it fits in with the rest of the paper.
Identify any redundancies (e.g. repeated figures or repeated text), if there are any, decide where in the paper things should be cut.
Identify where we can save space, and be more concise without weakening the message of the text.
Fix any remaining errors as before:
"""
    + error_list
)


# QUALITY-TRIGGERED REFINEMENT
quality_check_system_msg = """You are a strict scientific manuscript quality checker.
Your task is only to decide whether the supplied manuscript section requires
another editing pass.

Do not rewrite the section.
Do not suggest stylistic changes unless they materially improve correctness,
clarity, grounding, or scientific precision.

A section should be marked for refinement only if there is a concrete issue
such as:
- an unsupported or invented experimental claim,
- a numerical result not supported by the experiment artifacts,
- confusion between experimental runs and method/baseline comparisons,
- an invalid or invented figure reference,
- an invalid citation or citation key,
- duplicated or contradictory content,
- a clear LaTeX problem,
- an important omission relative to the section's scientific purpose,
- severe verbosity, ambiguity, or poor organization that weakens the paper.

Minor wording preferences, punctuation choices, typography preferences, and
cosmetic rewrites are not sufficient reasons for another refinement pass.

Return JSON only."""

quality_check_prompt = """Evaluate the following manuscript section.

Section name:
{section}

Section guidance:
{tips}

TARGET SECTION:
\"\"\"
{section_text}
\"\"\"

OTHER MANUSCRIPT SECTIONS:
\"\"\"
{other_sections_text}
\"\"\"

EXPERIMENT NOTES:
\"\"\"
{notes_text}
\"\"\"

Decide whether the TARGET SECTION needs another Aider refinement pass.

Check for concrete problems only, especially:

- numerical results or scientific claims that are not supported by the
  experiment notes;
- confusion between experimental runs, methods, configurations, and baselines;
- contradictions with other manuscript sections;
- substantial repetition of material already explained elsewhere in the paper;
- repeated presentation of the same result or figure without scientific need;
- invalid, invented, or poorly integrated figure references;
- invalid or unsupported citations;
- an important omission relative to the section's scientific purpose;
- severe verbosity, ambiguity, or organization problems that weaken the
  scientific argument;
- clear LaTeX or structural problems.

Use the OTHER MANUSCRIPT SECTIONS only to detect cross-section consistency,
redundancy, and organization problems. Do not ask to rewrite correct material
merely because another wording would be stylistically preferable.

Use the EXPERIMENT NOTES as evidence for experimental claims and numerical
results. If the notes do not support a claim, flag it. Do not invent missing
evidence.

Minor wording preferences, punctuation, typography, or cosmetic rewrites are
not sufficient reasons for another refinement pass.

Return exactly one JSON object with:
- "needs_refinement": true or false
- "issues": a short list of concrete issues that must be fixed

If the section is already scientifically sound, grounded, non-redundant, and
complete enough, return "needs_refinement": false and an empty issues list.
"""


def _extract_section_text(tex_text, section):
    """Extract only the requested manuscript section for a cheap quality check."""

    if section == "Abstract":
        match = re.search(
            r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
            tex_text,
            re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    aliases = {
        "Conclusion": [
            "Conclusion",
            "Conclusions",
            "Conclusions and Future Work",
            "Conclusion and Future Work",
        ],
    }

    possible_titles = aliases.get(section, [section])

    section_matches = list(
        re.finditer(
            r"\\section\{([^}]*)\}",
            tex_text,
        )
    )

    for idx, match in enumerate(section_matches):
        title = match.group(1).strip()
        if title not in possible_titles:
            continue

        start = match.end()
        end = (
            section_matches[idx + 1].start()
            if idx + 1 < len(section_matches)
            else len(tex_text)
        )
        return tex_text[start:end].strip()

    return ""



def _collect_other_sections_text(
    tex_text,
    target_section,
):
    """Collect compact manuscript context excluding the target section."""

    sections = [
        "Abstract",
        "Introduction",
        "Related Work",
        "Background",
        "Method",
        "Experimental Setup",
        "Results",
        "Conclusion",
    ]

    chunks = []

    for section in sections:
        if section == target_section:
            continue

        section_text = _extract_section_text(
            tex_text,
            section,
        )

        if not section_text:
            continue

        # Keep enough context for redundancy/consistency checks without
        # sending the complete paper multiple times.
        max_chars = 5000
        if len(section_text) > max_chars:
            section_text = (
                section_text[:max_chars]
                + "\n[SECTION TRUNCATED FOR QUALITY CHECK]"
            )

        chunks.append(
            f"### {section}\n{section_text}"
        )

    return "\n\n".join(chunks)


def assess_section_quality(
    client,
    model,
    folder_name,
    section,
):
    """Use a small section-only LLM call to decide whether Aider refinement is needed."""

    template_path = osp.join(
        folder_name,
        "latex",
        "template.tex",
    )

    with open(
        template_path,
        "r",
        encoding="utf-8",
    ) as f:
        tex_text = f.read()

    section_text = _extract_section_text(
        tex_text,
        section,
    )

    if not section_text:
        return True, [
            f"The {section} section could not be found or is empty."
        ]

    other_sections_text = _collect_other_sections_text(
        tex_text,
        section,
    )

    notes_path = osp.join(
        folder_name,
        "notes.txt",
    )

    if osp.exists(notes_path):
        with open(
            notes_path,
            "r",
            encoding="utf-8",
        ) as f:
            notes_text = f.read()
    else:
        notes_text = (
            "No notes.txt file was available. "
            "Do not assume unsupported experimental evidence."
        )

    # Avoid sending an unbounded notes file while retaining the experiment
    # history, results, and figure descriptions needed for grounding.
    max_notes_chars = 30000
    if len(notes_text) > max_notes_chars:
        notes_text = (
            notes_text[:max_notes_chars]
            + "\n[NOTES TRUNCATED FOR QUALITY CHECK]"
        )

    try:
        response, _ = get_response_from_llm(
            quality_check_prompt.format(
                section=section,
                tips=per_section_tips[section],
                section_text=section_text,
                other_sections_text=other_sections_text,
                notes_text=notes_text,
            ),
            client=client,
            model=model,
            system_message=quality_check_system_msg,
            msg_history=[],
            temperature=0.0,
        )

        result = extract_json_between_markers(response)

        if result is None:
            print(
                f"Quality check for {section} could not be parsed. "
                "Keeping the section unchanged."
            )
            return False, []

        needs_refinement = bool(
            result.get(
                "needs_refinement",
                False,
            )
        )

        issues = result.get(
            "issues",
            [],
        )

        if not isinstance(issues, list):
            issues = [str(issues)]

        return needs_refinement, issues

    except Exception as e:
        print(
            f"Quality check failed for {section}: {e}. "
            "Keeping the section unchanged."
        )
        return False, []


def refine_section_if_needed(
    coder,
    client,
    model,
    folder_name,
    section,
):
    """Refine a section only when the section-only checker finds concrete issues."""

    needs_refinement, issues = assess_section_quality(
        client,
        model,
        folder_name,
        section,
    )

    if not needs_refinement:
        print(
            f"Quality check passed for {section}; "
            "skipping refinement."
        )
        return False

    issue_text = "\n".join(
        f"- {issue}"
        for issue in issues
    )

    print(
        f"Quality check flagged {section} for refinement:"
    )
    print(issue_text)

    prompt = f"""Refine only the {section} section of `latex/template.tex`.

A lightweight scientific quality check identified these concrete issues:

{issue_text}

Fix only the identified issues.

If an issue concerns redundancy with another section, remove or shorten the
redundant material in THIS section while preserving the scientifically
appropriate material in the other section.

If an issue concerns an unsupported numerical or scientific claim, correct or
remove that claim using only evidence already present in `notes.txt`, the
completed experiment outputs, and the existing manuscript.

If an issue concerns figures, keep only scientifically useful figures and make
sure each retained figure is naturally introduced and discussed in the text.

Preserve correct scientific content, experimental results, citations, figures,
labels, and LaTeX structure that do not need changing.

Do not make cosmetic rewrites merely for style.
Do not introduce new claims, metrics, figures, citations, or experiments.
Do not modify other manuscript sections.

Use *SEARCH/REPLACE* blocks to perform the edit.
"""

    coder.run(prompt)
    return True


# CITATION HELPERS
citation_system_msg = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have already written an initial draft of the paper and now you are looking to add missing citations to related papers throughout the paper.
The related work section already has some initial comments on which papers to add and discuss.

Focus on completing the existing write-up and do not add entirely new elements unless necessary.
Ensure every point in the paper is substantiated with sufficient evidence.
Feel free to add more cites to a particular point if there is only one or two references.
Ensure no paper is cited without a corresponding reference in the `references.bib` file.
Ensure each paragraph of the related work has sufficient background, e.g. a few papers cited.
You will be given access to a literature search API, only add citations that you have found using the API.
Aim to discuss a broad range of relevant papers, not just the most popular ones.
Make sure not to copy verbatim from prior literature to avoid plagiarism.

You will be prompted to give a precise description of where and how to add the cite, and a search query for the paper to be cited.
Finally, you will select the most relevant cite from the search results (top 10 results will be shown).
You will have {total_rounds} rounds to add to the references, but do not need to use them all.

DO NOT ADD A CITATION THAT ALREADY EXISTS!"""

citation_first_prompt = '''Round {current_round}/{total_rounds}:

You have written this LaTeX draft so far:

"""
{draft}
"""

Identify the most important citation that you still need to add, and the query to find the paper.

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the paper and identify where citations should be added.
If no more citations are needed, add "No more citations needed" to your thoughts.
Do not add "No more citations needed" if you are adding citations this round.

In <JSON>, respond in JSON format with the following fields:
- "Description": A precise description of the required edit, along with the proposed text and location where it should be made.
- "Query": The search query to find the paper (e.g. attention is all you need).

Ensure the description is sufficient to make the change without further context. Someone else will make the change.
The query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''

citation_second_prompt = """Search has recovered the following articles:

{papers}

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the search results and identify which citation best fits your paper and the location is to be added at.
If none are appropriate, add "Do not add any" to your thoughts.

In <JSON>, respond in JSON format with the following fields:
- "Selected": A list of the indices of the selected papers to be cited, e.g. "[0, 1]". Can be "[]" if no papers are selected. This must be a string.
- "Description": Update the previous description of the required edit if needed. Ensure that any cites precisely match the name in the bibtex!!!

Do not select papers that are already in the `references.bib` file at the top of the draft, or if the same citation exists under a different name.
This JSON will be automatically parsed, so ensure the format is precise."""


def get_citation_aider_prompt(
        client, model, draft, current_round, total_rounds, engine="semanticscholar"
) -> Tuple[Optional[str], bool]:
    msg_history = []
    try:
        text, msg_history = get_response_from_llm(
            citation_first_prompt.format(
                draft=draft, current_round=current_round, total_rounds=total_rounds
            ),
            client=client,
            model=model,
            system_message=citation_system_msg.format(total_rounds=total_rounds),
            msg_history=msg_history,
        )
        if "No more citations needed" in text:
            print("No more citations needed.")
            return None, True

        ## PARSE OUTPUT
        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        query = json_output["Query"]
        papers = search_for_papers(query, engine=engine)
    except Exception as e:
        print(f"Error: {e}")
        return None, False

    if papers is None:
        print("No papers found.")
        return None, False

    paper_strings = []
    for i, paper in enumerate(papers):
        paper_strings.append(
            """{i}: {title}. {authors}. {venue}, {year}.\nAbstract: {abstract}""".format(
                i=i,
                title=paper["title"],
                authors=paper["authors"],
                venue=paper["venue"],
                year=paper["year"],
                abstract=paper["abstract"],
            )
        )
    papers_str = "\n\n".join(paper_strings)

    try:
        text, msg_history = get_response_from_llm(
            citation_second_prompt.format(
                papers=papers_str,
                current_round=current_round,
                total_rounds=total_rounds,
            ),
            client=client,
            model=model,
            system_message=citation_system_msg.format(total_rounds=total_rounds),
            msg_history=msg_history,
        )
        if "Do not add any" in text:
            print("Do not add any.")
            return None, False
        ## PARSE OUTPUT
        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        desc = json_output["Description"]
        selected_papers = json_output["Selected"]
        selected_papers = str(selected_papers)

        # convert to list
        if selected_papers != "[]":
            selected_papers = list(map(int, selected_papers.strip("[]").split(",")))
            assert all(
                [0 <= i < len(papers) for i in selected_papers]
            ), "Invalid paper index"
            bibtexs = []
            for i in selected_papers:
                paper = papers[i]
                if "citationStyles" in paper and paper["citationStyles"].get("bibtex"):
                    bibtexs.append(paper["citationStyles"]["bibtex"])
                else:
                    key = f"openalex_{current_round}_{i}"
                    bibtexs.append(
                        f"@article{{{key},\n"
                        f"  title = {{{paper.get('title', 'Unknown title')}}},\n"
                        f"  author = {{{paper.get('authors', 'Unknown authors')}}},\n"
                        f"  journal = {{{paper.get('venue', 'Unknown venue')}}},\n"
                        f"  year = {{{paper.get('year', 'Unknown year')}}}\n"
                        f"}}"
                    )
            bibtex_string = "\n".join(bibtexs)
        else:
            return None, False

    except Exception as e:
        print(f"Error: {e}")
        return None, False

    # Add citation to draft
    aider_format = '''The following citations have just been added to the end of the `references.bib` file definition at the top of the file:
"""
{bibtex}
"""
You do not need to add them yourself.
ABSOLUTELY DO NOT ADD IT AGAIN!!!

Make the proposed change to the draft incorporating these new cites:
{description}

Use your judgment for whether these should be cited anywhere else.
Make sure that any citation precisely matches the name in `references.bib`. Change its name to the correct name in the bibtex if needed.
Ensure the citation is well-integrated into the text.'''

    aider_prompt = (
            aider_format.format(bibtex=bibtex_string, description=desc)
            + r"""\n You must use \cite or \citet to reference papers, do not manually type out author names."""
    )
    return aider_prompt, False





# PERFORM WRITEUP
def perform_writeup(
    idea,
    folder_name,
    coder,
    cite_client,
    cite_model,
    num_cite_rounds=20,
    engine="openalex",
):
    # CURRENTLY ASSUMES LATEX
    abstract_prompt = f"""We've provided the `latex/template.tex` file to the project. We will be filling it in section by section.

First, please fill in the "Title" and "Abstract" sections of the writeup.

Some tips are provided below:
{per_section_tips["Abstract"]}

The experiments have already been completed. Use `notes.txt`, the generated figures,
and the completed run outputs as the source of truth for experimental claims.

Do not invent results, metrics, figures, comparisons, or statistical significance.
Do not refer to internal run directory names in the scientific prose.

Before every paragraph, please include a brief description of what you plan to write in that paragraph in a comment.

Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform these edits.
"""
    coder.run(abstract_prompt)

    for section in [
        "Introduction",
        "Background",
        "Method",
        "Experimental Setup",
        "Results",
        "Conclusion",
    ]:
        section_prompt = f"""
    Please fill in the {section} of the writeup.

    Some tips are provided below:

    {per_section_tips[section]}

    Use `notes.txt`, the generated figures, and the completed experimental
    outputs as the source of truth for the experiment and results.

    `notes.txt` contains descriptions of the generated figures and a recommended
    manuscript section for each figure.

    When writing the current section:

    - include only figures whose recommended manuscript section is appropriate
    for this section;
    - use the exact generated filename;
    - insert included figures using a proper LaTeX figure environment;
    - provide a concise, scientifically meaningful caption;
    - give each figure a unique LaTeX label;
    - refer to the figure naturally in the surrounding scientific text.

    Use the scientific purpose described in `notes.txt` to decide how each
    figure should be discussed.

    Do not invent figure filenames, figures, experimental results, metrics,
    comparisons, or interpretations that are not supported by the completed
    experiment artifacts.

    Do not treat internal run names, directory names, filenames, or other
    pipeline artifacts as scientific concepts in the manuscript.

    Be sure to use \\cite or \\citet where relevant, referring to the works
    provided in the file.

    Do not cite anything that is not already in `references.bib`.
    Do not add any new entries to this during this stage.

    Keep experimental findings, numerical comparisons, and result figures in
    the scientifically appropriate section.

    In this pass, do not reference anything in later sections of the paper.

    Before every paragraph, include a brief description of what you plan to
    write in that paragraph in a LaTeX comment.

    Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform
    these edits.
    """

        coder.run(section_prompt)

    # ============================================================
    # SKETCH RELATED WORK — Sakana-style
    # ============================================================

    section_prompt = f"""
Please fill in the Related Work of the writeup.

Some tips are provided below:

{per_section_tips["Related Work"]}

For this section, very briefly sketch out the structure of the section,
and clearly indicate what papers you intend to include.

Do this all in LaTeX comments using %.

The related work should be concise and should focus on the most relevant work.

Do not modify `references.bib` to add any new citations.
The bibliography will be filled in during the citation-search stage.

Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform
these edits.
"""

    coder.run(section_prompt)

    # ============================================================
    # ADD CITATIONS — Sakana-style loop
    # ============================================================
    
    citation_failures = 0
    for round_idx in range(num_cite_rounds):
        template_path = osp.join(
            folder_name,
            "latex",
            "template.tex",
        )

        with open(
            template_path,
            "r",
            encoding="utf-8",
        ) as f:
            draft = f.read()

        prompt, done = get_citation_aider_prompt(
            cite_client,
            cite_model,
            draft,
            round_idx,
            num_cite_rounds,
            engine=engine,
        )

        if done:
            break

        if prompt is None:
            citation_failures += 1

            print(
                f"Citation search failed or produced no usable citation "
                f"({citation_failures}/3 consecutive failures)."
            )

            if citation_failures >= 3:
                print(
                    "Citation search failed for 3 consecutive rounds. "
                    "Stopping citation search and continuing the writeup."
                )
                break

            continue

        citation_failures = 0

        bibtex_string = prompt.split('"""')[1]

        search_str = r"\end{filecontents}"

        draft = draft.replace(
            search_str,
            f"{bibtex_string}{search_str}",
        )

        with open(
            template_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(draft)

        coder.run(prompt)

    # ============================================================
    # QUALITY-TRIGGERED FINAL REFINEMENT
    # ============================================================

    # The complete draft now exists and citations have been added.
    # Check each section using only that section's text. Aider is invoked
    # only when the checker identifies a concrete scientific, grounding,
    # structural, citation, figure, or LaTeX issue.
    for section in [
        "Abstract",
        "Related Work",
        "Introduction",
        "Background",
        "Method",
        "Experimental Setup",
        "Results",
        "Conclusion",
    ]:
        refine_section_if_needed(
            coder=coder,
            client=cite_client,
            model=cite_model,
            folder_name=folder_name,
            section=section,
        )

    generate_latex(
        coder,
        folder_name,
        f"{folder_name}/{idea['Name']}.pdf",
    )


if __name__ == "__main__":
    from aider.coders import Coder
    from aider.models import Model
    from aider.io import InputOutput
    import json

    parser = argparse.ArgumentParser(description="Perform writeup for a project")
    parser.add_argument("--folder", type=str)
    parser.add_argument("--no-writing", action="store_true", help="Only generate")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="openalex",
        choices=["semanticscholar", "openalex"],
        help="Scholar engine to use.",
    )
    args = parser.parse_args()
    client, client_model = create_client(args.model)
    print("Make sure you cleaned the Aider logs if re-generating the writeup!")
    folder_name = args.folder
    idea_name = osp.basename(folder_name)
    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    model = args.model
    writeup_file = osp.join(folder_name, "latex", "template.tex")
    ideas_file = osp.join(folder_name, "ideas.json")
    with open(ideas_file, "r") as f:
        ideas = json.load(f)
    for idea in ideas:
        if idea["Name"] in idea_name:
            print(f"Found idea: {idea['Name']}")
            break
    if idea["Name"] not in idea_name:
        raise ValueError(f"Idea {idea_name} not found")
    fnames = [exp_file, writeup_file, notes, vis_file]
    io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
    if args.model == "deepseek-coder-v2-0724":
        main_model = Model("deepseek/deepseek-coder")
    elif args.model == "llama3.1-405b":
        main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
    else:
        main_model = Model(model)
    coder = Coder.create(
        main_model=main_model,
        fnames=fnames,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
    )
    if args.no_writing:
        generate_latex(coder, args.folder, f"{args.folder}/test.pdf")
    else:
        try:
            perform_writeup(idea, folder_name, coder, client, client_model, engine=args.engine)
        except Exception as e:
            print(f"Failed to perform writeup: {e}")