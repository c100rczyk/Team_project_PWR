import numpy as np

@staticmethod
class top5eff():
    def find_top_5(self, distances, labels):
        distances = np.array(distances)
        sorted_index = np.argsort(distances)
        top_5_idx = sorted_index[:5]

        top_5_distances = distances[top_5_idx]
        top_5_labels = [labels[i] for i in top_5_idx]

        return top_5_distances, top_5_labels

    def give_efficiency(self, label_camera, top5_labels, top5_dist):
        top1 = 1.0 * bool(label_camera == top5_labels[0])
        top2 = 0.98 * bool(label_camera == top5_labels[1])
        top3 = 0.96 * bool(label_camera == top5_labels[2])
        top4 = 0.94 * bool(label_camera == top5_labels[3])
        top5 = 0.92 * bool(label_camera == top5_labels[4])

        sum = top1 + top2 + top3 + top4 + top5

        return sum  # return one of top1-top5 or 0 if there is no good prediction in top5