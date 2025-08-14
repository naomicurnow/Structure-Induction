
import argparse
import numpy as np
import data_processing, search, evaluation
from graphs_and_forms import PartitionForm, OrderForm, ChainForm, RingForm, HierarchyForm, TreeForm, GridForm, CylinderForm
import sys
from pathlib import Path
import logging
from datetime import datetime
from plotting import plot_all, plot_best
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE  # repo root
sys.path.append(str(ROOT))

logger = logging.getLogger(__name__)

def main():
    main_log, imp_log = setup_logging()
    logger.info("Logging to %s", main_log)
    logger.info("Important log at %s", imp_log)

    parser = argparse.ArgumentParser(description="Run Kemp & Tenenbaum structural form discovery.")
    parser.add_argument("--file", type=str, help="File name (e.g., animals.mat).")
    parser.add_argument("--dataset", type=str, choices=["planets", "psychometric", "fmri"],
                        help="Load a built-in dataset.")
    parser.add_argument("--use_similarity", action="store_true",
                        help="Use similarity matrix instead of raw features.")
    parser.add_argument("--plot_entity_labels", action="store_true",
                        help="Add entity labels to the plot (if false, adds black dots on the entities).")
    args = parser.parse_args()

    if not args.file and not args.dataset:
        raise ValueError("Provide --file or --dataset.")
    
    if args.dataset:
        if args.dataset == "planets":
            df = data_processing.load_planet_data()
        if args.dataset == "psychometric":
            df = data_processing.load_psychometric_data()
        if args.dataset == "fmri":
            df = data_processing.load_fmri_data()
    else:
        if args.use_similarity:
            df = data_processing.load_sim_data(args.file)
        else:
            df = data_processing.load_feature_data(args.file)

    # make data matrix
    if args.use_similarity:
        Sim = data_processing.dataframe_to_matrix(df)
        Sim_proc = data_processing.preprocess_similarity_matrix(Sim)
        data = Sim_proc
        is_similarity = True
        n_entities = Sim.shape[0]
    else:
        D = data_processing.dataframe_to_matrix(df)
        D_proc = data_processing.centre_and_scale_features(D)
        data = D_proc
        is_similarity = False
        n_entities = D.shape[0]

    forms = [PartitionForm(), OrderForm(), ChainForm(), RingForm(), HierarchyForm(), TreeForm(), GridForm(), CylinderForm()]

    results = search.search_over_forms(forms, data, n_entities, is_similarity)
    scores = {name: score for name, (_, score) in results.items()}
    norm = evaluation.relative_scores(scores)
    best_form = evaluation.select_best_form(scores)
    post_probs = evaluation.softmax_probabilities(scores)

    logger.important("Scores (raw):")
    for name, (graph, score) in results.items():
        logger.important(f"  {name}: {score:.3f}")

    logger.important("")

    logger.important("Relative scores:")
    for name, val in norm.items():
        logger.important(f"  {name}: {val:.3f}")

    logger.important("")

    logger.important(f"Best form: {best_form}")
    logger.important("Softmax probabilities:")
    for name, p in post_probs.items():
        logger.important(f"  {name}: {p:.3f}")

    best_graph, _ = results[best_form]

    logger.important("")
    
    logger.important("*** Best graph structure: ***")
    logger.important(best_graph)

    logger.important("")

    logger.important("*** All graphs: ***")

    for i, (name, (cg, score)) in enumerate(sorted(results.items(), key=lambda kv: kv[1][1], reverse=True), 1):
        logger.important(f"Graph #{i} {name} (score={score:.2f})")
        logger.important(cg)     
        logger.important("")   

    # plotting

    if args.plot_entity_labels:
        entity_names = df['name'].tolist()
    else:
        entity_names = None

    graphs = [(name, results[name][0]) for name in results]
    
    base_tag = args.dataset if args.dataset else Path(args.file).stem
    plot_folder = Path(ROOT / "plots" / base_tag)
    plot_folder.mkdir(parents=True, exist_ok=True)

    plot_all(graphs, plot_folder / "all_graphs_v1.png", scores, plot_type='fixed', entity_names=entity_names)
    plot_all(graphs, plot_folder / "all_graphs_v2.png", scores, plot_type='unfixed', entity_names=entity_names)
    plot_all(graphs, plot_folder / "all_graphs_v3.png", scores, plot_type='non_overlapping', entity_names=entity_names)

    fig = plot_best(best_graph, entity_names=entity_names)      
    fig.show()


def setup_logging():
    # one run = one unique file pair
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    main_log_path = log_dir / f"run-{run_id}.log"
    imp_log_path  = log_dir / f"run-{run_id}-important.log"

    IMPORTANT = 25
    logging.addLevelName(IMPORTANT, "IMPORTANT")

    def important(self, msg, *args, **kwargs):
        # mark record as important so the filter can tee it
        extra = kwargs.setdefault("extra", {})
        extra["important"] = True
        if self.isEnabledFor(IMPORTANT):
            self._log(IMPORTANT, msg, args, **kwargs)
    logging.Logger.important = important

    class StarFilter(logging.Filter):
        def filter(self, record):
            # add 'star' field used by the formatter
            star = " ⭐" if getattr(record, "important", False) else ""
            setattr(record, "star", star)
            return True

    class ImportantOnly(logging.Filter):
        def filter(self, record):
            return getattr(record, "important", False)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()  # avoid dupes if setup_logging() is called multiple times

    # main file: everything, with a visible star on important lines
    main_fh = logging.FileHandler(main_log_path, encoding="utf-8")
    main_fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s]%(star)s %(name)s: %(message)s"
    ))
    main_fh.addFilter(StarFilter())
    root.addHandler(main_fh)

    # important-only file: just the flagged lines
    imp_fh = logging.FileHandler(imp_log_path, encoding="utf-8")
    imp_fh.setFormatter(logging.Formatter(
        "%(asctime)s [IMPORTANT] %(name)s: %(message)s"
    ))
    imp_fh.addFilter(ImportantOnly())
    root.addHandler(imp_fh)

    # echo to console too
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s]%(star)s %(message)s"))
    ch.addFilter(StarFilter())
    root.addHandler(ch)

    return main_log_path, imp_log_path



if __name__ == "__main__":
    main()
