import json
import Levenshtein

def evaluate_extraction_metrics_per_sample(prediction_str, ground_truth_str):
    """
    Computes Schema Compliance, Field-level NED, and Field-level ANLS 
    for a document entity extraction task.
    """
    # Metric 1: Structural / Schema Compliance Tracks if the output string is a valid, uncorrupted JSON object.
    try:
        pred_dict = json.loads(prediction_str.strip())
        gt_dict = json.loads(ground_truth_str.strip())
        schema_compliance = 1.0
    except Exception:
        # if JSON parsing fails due to broken braces or structural errors
        return {
            "schema_compliance": 0.0,
            "overall_ned": 0.0,
            "overall_anls": 0.0,
            "field_metrics": {}
        }
        
    field_metrics = {}
    total_fields = 0
    running_ned = 0.0
    running_anls = 0.0
    
    # evaluate performance across all keys expected in the schema
    for key in gt_dict.keys():
        total_fields += 1
        gt_val = str(gt_dict[key]).strip().upper()
        # handle cases where the model completely misses a required key
        pred_val = str(pred_dict.get(key, "")).strip().upper()
        
        # calculates raw Levenshtein distance
        distance = Levenshtein.distance(pred_val, gt_val)
        max_len = max(len(pred_val), len(gt_val))
        
        # Metric 2: Normalized Edit Distance (NED)
        if max_len == 0:
            ned = 1.0
        else:
            ned = 1.0 - (distance / max_len)
            
        # Metric 3: Average Normalized Levenshtein Similarity (ANLS) for Document AI: if similarity drops below 0.5, 
        # it treats the entire extraction as a complete hallucination (0.0).
        anls_threshold = 0.5
        anls = ned if ned >= anls_threshold else 0.0
        
        field_metrics[key] = {
            "ground_truth": gt_val,
            "prediction": pred_val,
            "ned": round(ned, 4),
            "anls": round(anls, 4)
        }
        
        running_ned += ned
        running_anls += anls

    return {
        "schema_compliance": schema_compliance,
        "overall_ned": round(running_ned / total_fields, 4),
        "overall_anls": round(running_anls / total_fields, 4),
        "field_metrics": field_metrics
    }