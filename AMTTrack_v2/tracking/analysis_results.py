import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
import argparse

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    # parser.add_argument('--dataset', type=str, default='coesot', help='coesot or fe108 or visevent or eed.')
    # # parser.add_argument('--parameter_name', type=str, default='ceutrack_coesot', help='coesot or fe108 or visevent or eed.')
    # args = parser.parse_args()

    trackers = []
    # dataset_name = args.dataset      # fe108
    dataset_name = 'fe108'
    parameter_name = f'amttrack_{dataset_name}'

    """amttrack"""
    trackers.extend(trackerlist(name='amttrack', parameter_name=parameter_name, dataset_name=dataset_name,
                                run_ids=None, display_name='AMTTrack'))
    
    dataset = get_dataset(dataset_name)
    print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))