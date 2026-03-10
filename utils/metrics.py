import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import sed_eval
except ImportError:
    print("Warning: sed_eval not installed. Run: pip install sed_eval")

try:
    from psds_eval import PSDSEval
except ImportError:
    print("Warning: psds_eval not installed. Run: pip install psds_eval")


class SEDMetricsCalculator:
    def __init__(self, class_names, segment_length: float = 1.0):
        self.class_names = class_names
        self.segment_length = segment_length

    def calculate_all_metrics(self,
                              predictions_prob: np.ndarray,
                              targets: np.ndarray,
                              dataset,
                              thresholds=None):
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)

        ground_truth_df, metadata_df = self._prepare_psds_ground_truth(dataset)
        ground_truth_df = self._merge_overlapping_events(ground_truth_df)

        psds1_eval = self._create_psds_evaluator(
            ground_truth_df, metadata_df,
            dtc_threshold=0.7, gtc_threshold=0.7,
            alpha_ct=0.0, alpha_st=1.0, max_efpr=100
        )
        psds2_eval = self._create_psds_evaluator(
            ground_truth_df, metadata_df,
            dtc_threshold=0.1, gtc_threshold=0.1,
            cttc_threshold=0.3, alpha_ct=0.5, alpha_st=1.0, max_efpr=100
        )

        for thr in tqdm(thresholds, desc="Evaluating thresholds"):
            detections_df = self._prepare_psds_detections(predictions_prob, dataset, thr)
            if not detections_df.empty:
                detections_df = self._merge_overlapping_events(detections_df)
            try:
                psds1_eval.add_operating_point(detections_df)
                psds2_eval.add_operating_point(detections_df)
            except Exception as e:
                print(f"Warning at threshold {thr:.2f}: {e}")

        try:
            psds1_result = psds1_eval.psds(alpha_st=1.0, alpha_ct=0.0, max_efpr=100)
            psds1_score = float(psds1_result.value) if hasattr(psds1_result, "value") else float(psds1_result)
        except Exception as e:
            print(f"Warning: PSDS1 calculation failed: {e}")
            psds1_score = 0.0

        try:
            psds2_result = psds2_eval.psds(alpha_st=1.0, alpha_ct=0.5, max_efpr=100)
            psds2_score = float(psds2_result.value) if hasattr(psds2_result, "value") else float(psds2_result)
        except Exception as e:
            print(f"Warning: PSDS2 calculation failed: {e}")
            psds2_score = 0.0

        predictions_binary = (predictions_prob >= 0.5).astype(int)
        reference_events, estimated_events = self._convert_to_events(
            predictions_binary, targets, dataset
        )

        empty_metrics = {
            "f1_micro": 0.0, "precision_micro": 0.0,
            "recall_micro": 0.0, "f1_macro": 0.0, "error_rate": 0.0,
        }
        event_based_metrics = dict(empty_metrics)
        segment_based_metrics = dict(empty_metrics)

        if reference_events and estimated_events:
            event_based_metrics = self._calculate_event_based_metrics(
                reference_events, estimated_events
            )
            segment_based_metrics = self._calculate_segment_based_metrics(
                reference_events, estimated_events
            )

        return {
            "event_based": event_based_metrics,
            "segment_based": segment_based_metrics,
            "psds1": psds1_score,
            "psds2": psds2_score,
        }

    def _merge_overlapping_events(self, detections_df: pd.DataFrame) -> pd.DataFrame:
        if detections_df.empty:
            return detections_df

        merged_data = []
        for (filename, event_label), group in detections_df.groupby(["filename", "event_label"]):
            group = group.sort_values("onset").reset_index(drop=True)
            current_onset = group.iloc[0]["onset"]
            current_offset = group.iloc[0]["offset"]

            for i in range(1, len(group)):
                next_onset = group.iloc[i]["onset"]
                next_offset = group.iloc[i]["offset"]
                if next_onset <= current_offset:
                    current_offset = max(current_offset, next_offset)
                else:
                    merged_data.append({
                        "filename": filename, "onset": current_onset,
                        "offset": current_offset, "event_label": event_label,
                    })
                    current_onset = next_onset
                    current_offset = next_offset

            merged_data.append({
                "filename": filename, "onset": current_onset,
                "offset": current_offset, "event_label": event_label,
            })

        return pd.DataFrame(merged_data)

    def _prepare_psds_ground_truth(self, dataset):
        gt_data = []
        durations = {}

        for sample in dataset.samples:
            filename = sample["subject_id"]
            onset = sample["start_time"]
            offset = sample["end_time"]
            event_label = sample["label"]

            gt_data.append({
                "filename": filename, "onset": onset,
                "offset": offset, "event_label": event_label,
            })

            if filename not in durations:
                durations[filename] = offset
            else:
                durations[filename] = max(durations[filename], offset)

        ground_truth_df = pd.DataFrame(gt_data)
        metadata_df = pd.DataFrame(
            [{"filename": fn, "duration": dur} for fn, dur in durations.items()]
        )
        return ground_truth_df, metadata_df

    def _prepare_psds_detections(self, predictions_prob, dataset, threshold):
        det_data = []
        for idx, sample in enumerate(dataset.samples):
            filename = sample["subject_id"]
            onset = sample["start_time"]
            offset = sample["end_time"]
            for class_idx, class_name in enumerate(self.class_names):
                if predictions_prob[idx][class_idx] >= threshold:
                    det_data.append({
                        "filename": filename, "onset": onset,
                        "offset": offset, "event_label": class_name,
                    })
        return pd.DataFrame(det_data)

    def _create_psds_evaluator(self, ground_truth_df, metadata_df,
                                dtc_threshold, gtc_threshold,
                                alpha_ct, alpha_st, max_efpr,
                                cttc_threshold=None):
        if cttc_threshold is not None:
            return PSDSEval(
                ground_truth=ground_truth_df, metadata=metadata_df,
                dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
                cttc_threshold=cttc_threshold,
            )
        return PSDSEval(
            ground_truth=ground_truth_df, metadata=metadata_df,
            dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        )

    def _convert_to_events(self, predictions, targets, dataset):
        reference_events = []
        estimated_events = []

        for idx, sample in enumerate(dataset.samples):
            subject_id = sample["subject_id"]
            start_time = sample["start_time"]
            end_time = sample["end_time"]

            for class_idx, class_name in enumerate(self.class_names):
                if targets[idx][class_idx] == 1:
                    reference_events.append({
                        "filename": subject_id, "event_label": class_name,
                        "onset": start_time, "offset": end_time,
                    })

            for class_idx, class_name in enumerate(self.class_names):
                if predictions[idx][class_idx] == 1:
                    estimated_events.append({
                        "filename": subject_id, "event_label": class_name,
                        "onset": start_time, "offset": end_time,
                    })

        return reference_events, estimated_events

    def _calculate_event_based_metrics(self, reference_events, estimated_events):
        try:
            event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
                event_label_list=self.class_names,
                t_collar=0.200,
                percentage_of_length=0.2,
            )
            files = list(set([e["filename"] for e in reference_events + estimated_events]))
            for filename in files:
                ref_file = [e for e in reference_events if e["filename"] == filename]
                est_file = [e for e in estimated_events if e["filename"] == filename]
                if ref_file or est_file:
                    event_based_metrics.evaluate(ref_file, est_file)

            results = event_based_metrics.results()
            f1_micro = results["overall"]["f_measure"].get("f_measure", 0.0)
            precision_micro = results["overall"]["f_measure"].get("precision", 0.0)
            recall_micro = results["overall"]["f_measure"].get("recall", 0.0)
            error_rate = results["overall"]["error_rate"].get("error_rate", 0.0)
            f1_values = [
                results["class_wise"][c]["f_measure"].get("f_measure", 0.0)
                for c in self.class_names if c in results["class_wise"]
            ]
            f1_macro = np.mean([v for v in f1_values if not np.isnan(v)]) if f1_values else 0.0

            return {
                "f1_micro": f1_micro if not np.isnan(f1_micro) else 0.0,
                "precision_micro": precision_micro if not np.isnan(precision_micro) else 0.0,
                "recall_micro": recall_micro if not np.isnan(recall_micro) else 0.0,
                "f1_macro": f1_macro if not np.isnan(f1_macro) else 0.0,
                "error_rate": error_rate if not np.isnan(error_rate) else 0.0,
            }
        except Exception as e:
            print(f"Warning: Event-based metrics failed: {e}")
            return {"f1_micro": 0.0, "precision_micro": 0.0,
                    "recall_micro": 0.0, "f1_macro": 0.0, "error_rate": 0.0}

    def _calculate_segment_based_metrics(self, reference_events, estimated_events):
        try:
            segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
                event_label_list=self.class_names,
                time_resolution=self.segment_length,
            )
            files = list(set([e["filename"] for e in reference_events + estimated_events]))
            for filename in files:
                ref_file = [e for e in reference_events if e["filename"] == filename]
                est_file = [e for e in estimated_events if e["filename"] == filename]
                if ref_file or est_file:
                    max_offset = max([e["offset"] for e in ref_file + est_file] or [0])
                    segment_based_metrics.evaluate(
                        ref_file, est_file, evaluated_length_seconds=max_offset,
                    )

            results = segment_based_metrics.results()
            f1_micro = results["overall"]["f_measure"].get("f_measure", 0.0)
            precision_micro = results["overall"]["f_measure"].get("precision", 0.0)
            recall_micro = results["overall"]["f_measure"].get("recall", 0.0)
            error_rate = results["overall"]["error_rate"].get("error_rate", 0.0)
            f1_values = [
                results["class_wise"][c]["f_measure"].get("f_measure", 0.0)
                for c in self.class_names if c in results["class_wise"]
            ]
            f1_macro = np.mean([v for v in f1_values if not np.isnan(v)]) if f1_values else 0.0

            return {
                "f1_micro": f1_micro if not np.isnan(f1_micro) else 0.0,
                "precision_micro": precision_micro if not np.isnan(precision_micro) else 0.0,
                "recall_micro": recall_micro if not np.isnan(recall_micro) else 0.0,
                "f1_macro": f1_macro if not np.isnan(f1_macro) else 0.0,
                "error_rate": error_rate if not np.isnan(error_rate) else 0.0,
            }
        except Exception as e:
            print(f"Warning: Segment-based metrics failed: {e}")
            return {"f1_micro": 0.0, "precision_micro": 0.0,
                    "recall_micro": 0.0, "f1_macro": 0.0, "error_rate": 0.0}
