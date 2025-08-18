
import argparse
import re
import sys
import logging
from pathlib import Path
import pandas as pd

import data_processing, search, evaluation
from graphs_and_forms import PartitionForm, OrderForm, ChainForm, RingForm, HierarchyForm, TreeForm, GridForm, CylinderForm
from plotting import plot_all, plot_best
from run_model import setup_logging

HERE = Path(__file__).resolve().parent
ROOT = HERE  # repo root
sys.path.append(str(ROOT))

logger = logging.getLogger(__name__)

def read_labels(labels_path: Path) -> list[str]:
    """Read one label per line from a text file (UTF-8)."""
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip() != ""]
    return labels

def parse_participant_id(path: Path) -> int | None:
    """
    Extract a participant id from filenames like: noun_behav_subj03.csv
    Returns an int (e.g., 3) or None if not found.
    """
    m = re.search(r"subj(\d+)", path.stem)
    if m:
        return int(m.group(1))
    return None

def run_one_participant(fp: Path, labels: list[str]):
    """
    Load a similarity CSV for one participant and run the full form search.
    Returns (best_form_name, scores_dict, post_probs_dict).
    """
    pid = parse_participant_id(fp)
    tag = str(pid)
    # if tag == '0': return 0, {}, {}
    logger.info("Participant %s: loading %s", f"{pid:02d}" if pid is not None else "NA", fp)

    df = pd.read_csv(fp)
    Sim = data_processing.dataframe_to_matrix(df)
    Sim_proc = data_processing.preprocess_similarity_matrix(Sim)
    data = Sim_proc
    is_similarity = True
    n_entities = Sim.shape[0]

    forms = [TreeForm()]

    results = search.search_over_forms(forms, data, n_entities, is_similarity)
    scores = {name: score for name, (_, score) in results.items()}
    norm = evaluation.relative_scores(scores)
    best_form = evaluation.select_best_form(scores)
    post_probs = evaluation.softmax_probabilities(scores)

    logger.important(f"[Participant {tag}] Scores (raw):")
    for name, (graph, score) in results.items():
        logger.important(f"  {name}: {score:.3f}")

    logger.important("")
    logger.important(f"[Participant {tag}] Relative scores:")
    for name, val in norm.items():
        logger.important(f"  {name}: {val:.3f}")

    logger.important("")
    logger.important(f"[Participant {tag}] Best form: {best_form}")
    logger.important(f"[Participant {tag}] Softmax probabilities:")
    for name, p in post_probs.items():
        logger.important(f"  {name}: {p:.3f}")

    best_graph, _ = results[best_form]

    logger.important("")
    logger.important(f"[Participant {tag}] *** Best graph structure: ***")
    logger.important(best_graph)
    logger.important("")

    logger.important(f"[Participant {tag}] *** All graphs: ***")
    for i, (name, (cg, score)) in enumerate(sorted(results.items(), key=lambda kv: kv[1][1], reverse=True), 1):
        logger.important(f"Graph #{i} {name} (score={score:.2f})")
        logger.important(cg)
        logger.important("")

    graphs = [(name, results[name][0]) for name in results]

    plot_folder = Path(ROOT / "plots" / "behav" / tag)
    plot_folder.mkdir(parents=True, exist_ok=True)

    entity_names = labels
    plot_all(graphs, plot_folder / "all_graphs_v1.png", scores, plot_type='fixed', entity_names=entity_names)
    plot_all(graphs, plot_folder / "all_graphs_v2.png", scores, plot_type='unfixed', entity_names=entity_names)
    plot_all(graphs, plot_folder / "all_graphs_v3.png", scores, plot_type='non_overlapping', entity_names=entity_names)
    logger.info("Participant %s: plots written to %s", f"{pid:02d}" if pid is not None else "NA", plot_folder)

    return best_form, scores, post_probs

def main():
    parser = argparse.ArgumentParser(description="Run Kemp & Tenenbaum structural form discovery on word similarity.")
    parser.add_argument("--pid", type=str, help="ID of the participant.")
    args = parser.parse_args()

    main_log, imp_log = setup_logging()
    logger.info("Logging to %s", main_log)
    logger.info("Important log at %s", imp_log)

    # behav_dir = ROOT / 'word_similarity' / 'data' / 'behav'
    # files = sorted(behav_dir.glob("noun_behav_subj*.csv"))
    # if not files:
    #     raise FileNotFoundError(f"No files matched in {behav_dir} with pattern: noun_behav_subj*.csv")
    # logger.info("Found %d participant files in %s.", len(files), behav_dir)

    if int(args.pid) < 10:
        ID = "0" + str(args.pid)
    elif int(args.pid) < 36:
        ID = str(args.pid)
    else:
        raise ValueError("Pid not within the required range -- 0 to 35 inclusive")

    data = ROOT / 'word_similarity' / 'data' / 'behav' / f'noun_behav_subj{ID}.csv'

    labels_path = ROOT / 'word_similarity' / 'data' / 'labels.txt'
    labels = read_labels(labels_path)
    logger.info("Loaded %d labels from %s", len(labels), labels_path) 

    run_one_participant(data, labels)

    # summary_rows = []
    # for i, fp in enumerate(files, 1):
    #     pid = parse_participant_id(fp)
    #     best_form, scores, post_probs = run_one_participant(fp, labels)
    #     summary_rows.append({
    #         "participant": pid,
    #         "file": fp.name,
    #         "best_form": best_form,
    #         **{f"score_{k}": v for k, v in scores.items()},
    #         **{f"prob_{k}": p for k, p in post_probs.items()},
    #     })

    # # write a CSV summary of best forms and scores
    # summary_df = pd.DataFrame(summary_rows)
    # summary_path = behav_dir / "participants_summary.csv"
    # summary_df.to_csv(summary_path, index=False)
    # logger.important(f"Summary written to: {summary_path}")

if __name__ == "__main__":
    main()