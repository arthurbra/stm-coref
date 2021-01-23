import argparse
import json
import os

from bfcr_model import BFCRModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicts coref-clusters in the given texts using the pretrained '
                                                 'BFCR_Span_Onto_STM-Model and exports the results to a file.')
    parser.add_argument('texts_fp', type=str,
                        help='Path to a Jsonlines-file, where each line is a json containing a text.')
    parser.add_argument('domains_fp', type=str, nargs='?',
                        help='Path to a Jsonlines-file, where each line is a json containing a domain.')
    parser.add_argument('predictions_dir', type=str, nargs='?',
                        help='Path to the directory where the predicted-clusters-file and standoff_annotations '
                             'will be saved to. (default: data/coref_predictions/)',
                        default=os.path.join(BFCRModel.DATA_DIR, 'coref_predictions'))
    parser.add_argument('--disable_saving_clusters_to_file', action='store_true',
                        help='Disables saving the predicted_clusters to a predicted_clusters.jsonlines.', default=False)
    parser.add_argument('--create_standoff_annotations', action='store_true', default=False,
                        help='Creates standoff_annotations (brat) for each text with the predicted clusters')

    args = parser.parse_args()
    with open(args.texts_fp) as file:
        texts = [json.loads(line) for line in file]

    if args.domains_fp:
        with open(args.domains_fp) as file:
            domains = [json.loads(line) for line in file]
    else:
        domains = None

    bfcr_model = BFCRModel()
    all_predicted_clusters = bfcr_model.predict(
        texts, domains, remove_predictions_file=True,
        create_standoff_annotations=args.create_standoff_annotations,
        standoff_annotations_dir=os.path.join(args.predictions_dir, 'standoff')
    )

    if not args.disable_saving_clusters_to_file:
        with open(os.path.join(args.predictions_dir, 'predicted_clusters.jsonlines'), 'w') as file:
            for predicted_clusters in all_predicted_clusters:
                file.write(json.dumps(predicted_clusters) + '\n')
