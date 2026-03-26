import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

GLOBAL_THRESHOLD = 0.295
MIN_PRECISION_CONSTRAINT = 0.20

def get_age_group(age):
    if age < 40:
        return "<40"
    elif age <= 54:
        return "40-54"
    elif age <= 64:
        return "55-64"
    elif age <= 74:
        return "65-74"
    else:
        return "75+"

def calculate_metrics(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = f1_score(y_true, preds, zero_division=0)
    mcc = matthews_corrcoef(y_true, preds)
    
    return {
        "recall": float(recall),
        "precision": float(precision),
        "specificity": float(specificity),
        "f1": float(f1),
        "mcc": float(mcc),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn)
    }

def find_optimal_thresholds(y_true, probs, thresholds=np.arange(0.01, 0.99, 0.005)):
    best_mcc_thresh = 0.5
    max_mcc = -1.0
    
    best_rec_thresh = 0.5
    max_rec = -1.0
    
    for t in thresholds:
        metrics = calculate_metrics(y_true, probs, t)
        
        if metrics["mcc"] > max_mcc:
            max_mcc = metrics["mcc"]
            best_mcc_thresh = t
            
        if metrics["precision"] >= MIN_PRECISION_CONSTRAINT and metrics["recall"] > max_rec:
            max_rec = metrics["recall"]
            best_rec_thresh = t
            
    # Fallback if no threshold meets precision constraint
    if max_rec == -1.0:
        best_rec_thresh = best_mcc_thresh
        
    return best_mcc_thresh, best_rec_thresh

def process_data(val_probs_file, val_labels_file, val_meta_file, 
                 test_probs_file, test_labels_file, test_meta_file, output_file):
    
    # 1. Load validation data
    logging.info("Loading validation data...")
    val_probs = pd.read_csv(val_probs_file)
    val_labels = pd.read_csv(val_labels_file)
    val_meta = pd.read_csv(val_meta_file)
    
    val_df = pd.concat([val_probs, val_labels, val_meta], axis=1)
    # Ensure standard column names: 'prob', 'y_true', 'age', 'gender'
    # Assuming columns might be named 'prob_cal', 'readmit_30', 'anchor_age', 'gender' etc.
    # We will rename assuming the first column of each file or specific known names
    # For robustness, we'll try to guess if they aren't provided explicitly.
    prob_col = 'prob_cal' if 'prob_cal' in val_df.columns else val_probs.columns[0]
    label_col = 'readmit_30' if 'readmit_30' in val_df.columns else ('y_true' if 'y_true' in val_df.columns else val_labels.columns[0])
    age_col = 'anchor_age' if 'anchor_age' in val_df.columns else ('age' if 'age' in val_df.columns else 'Age')
    gender_col = 'gender' if 'gender' in val_df.columns else 'Gender'
    
    # Define Subgroups
    val_df['age_group'] = val_df[age_col].apply(get_age_group)
    val_df['gender_group'] = val_df[gender_col].apply(lambda x: "Female" if str(x).upper() in ["F", "0", "FEMALE"] else "Male")
    
    subgroups = {
        "Age": list(val_df['age_group'].unique()),
        "Gender": list(val_df['gender_group'].unique())
    }
    
    # 2. Find optimal thresholds on Validation
    logging.info("Finding optimal subgroup thresholds on validation set...")
    subgroup_thresholds = {}
    
    for feature, groups in subgroups.items():
        col = 'age_group' if feature == "Age" else 'gender_group'
        for group in groups:
            mask = val_df[col] == group
            subset = val_df[mask]
            
            n_samples = len(subset)
            if n_samples < 100:
                logging.warning(f"Validation subgroup {feature}={group} has only {n_samples} samples (<100).")
                
            y_sub = subset[label_col].values
            p_sub = subset[prob_col].values
            
            t_mcc, t_rec = find_optimal_thresholds(y_sub, p_sub)
            subgroup_thresholds[f"{feature}_{group}"] = {
                "opt_mcc_thresh": t_mcc,
                "opt_rec_thresh": t_rec,
                "val_n": n_samples
            }
            logging.info(f"{feature}={group}: Opt MCC thresh={t_mcc:.4f}, Opt Recall thresh={t_rec:.4f}")

    # 3. Load test data
    logging.info("Loading test data...")
    test_probs = pd.read_csv(test_probs_file)
    test_labels = pd.read_csv(test_labels_file)
    test_meta = pd.read_csv(test_meta_file)
    
    test_df = pd.concat([test_probs, test_labels, test_meta], axis=1)
    test_df['age_group'] = test_df[age_col].apply(get_age_group)
    test_df['gender_group'] = test_df[gender_col].apply(lambda x: "Female" if str(x).upper() in ["F", "0", "FEMALE"] else "Male")
    
    # 4. Apply thresholds to Test Set
    logging.info("Evaluating thresholds on test set...")
    report = []
    
    for feature, groups in subgroups.items():
        col = 'age_group' if feature == "Age" else 'gender_group'
        for group in groups:
            mask = test_df[col] == group
            subset = test_df[mask]
            
            n_samples = len(subset)
            if n_samples < 100:
                logging.warning(f"Test subgroup {feature}={group} has only {n_samples} samples (<100).")
                
            if n_samples == 0:
                continue
                
            y_sub = subset[label_col].values
            p_sub = subset[prob_col].values
            
            # Global baseline
            mets_global = calculate_metrics(y_sub, p_sub, GLOBAL_THRESHOLD)
            
            # Subgroup Specific metrics
            t_mcc = subgroup_thresholds[f"{feature}_{group}"]["opt_mcc_thresh"]
            mets_mcc = calculate_metrics(y_sub, p_sub, t_mcc)
            
            t_rec = subgroup_thresholds[f"{feature}_{group}"]["opt_rec_thresh"]
            mets_rec = calculate_metrics(y_sub, p_sub, t_rec)
            
            report.append({
                "Feature": feature,
                "Subgroup": group,
                "Test_N": n_samples,
                "Global_Thresh": GLOBAL_THRESHOLD,
                "Opt_MCC_Thresh": t_mcc,
                "Opt_Rec_Thresh": t_rec,
                "Global_Metrics": mets_global,
                "Opt_MCC_Metrics": mets_mcc,
                "Opt_Rec_Metrics": mets_rec,
                "Improvement_MCC_vs_Global": mets_mcc["mcc"] - mets_global["mcc"],
                "Improvement_Recall_vs_Global": mets_mcc["recall"] - mets_global["recall"] # Typically comparing MCC opt
            })

    # Print summary
    print("\n" + "="*80)
    print(f"{'Subgroup':<20} | {'Glob Thr':<10} | {'Opt Thr':<10} | {'MCC Glob -> Opt':<20} | {'Rec Glob -> Opt':<20}")
    print("-" * 80)
    
    improvements_mcc = []
    improvements_rec = []
    
    for r in report:
        gname = f"{r['Feature']}={r['Subgroup']}"
        mcc_g = r['Global_Metrics']['mcc']
        mcc_o = r['Opt_MCC_Metrics']['mcc']
        rec_g = r['Global_Metrics']['recall']
        rec_o = r['Opt_MCC_Metrics']['recall']
        
        diff_mcc = mcc_o - mcc_g
        diff_rec = rec_o - rec_g
        
        improvements_mcc.append(diff_mcc)
        improvements_rec.append(diff_rec)
        
        mcc_str = f"{mcc_g:.3f} -> {mcc_o:.3f} ({diff_mcc:+.3f})"
        rec_str = f"{rec_g:.3f} -> {rec_o:.3f} ({diff_rec:+.3f})"
        
        print(f"{gname:<20} | {r['Global_Thresh']:<10.3f} | {r['Opt_MCC_Thresh']:<10.3f} | {mcc_str:<20} | {rec_str:<20}")

    print("="*80)
    print(f"Average MCC Improvement: {np.mean(improvements_mcc):+.4f}")
    print(f"Average Recall Improvement: {np.mean(improvements_rec):+.4f}")
    
    # Save results
    output_data = {
        "global_baseline_threshold": GLOBAL_THRESHOLD,
        "subgroup_thresholds": subgroup_thresholds,
        "test_results": report,
        "summary": {
            "avg_mcc_improvement": float(np.mean(improvements_mcc)),
            "avg_recall_improvement": float(np.mean(improvements_rec))
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    logging.info(f"Results saved heavily into {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subgroup-specific Threshold Optimization for Fairness")
    parser.add_argument("--val-probs", required=True, help="CSV file containing validation probabilities")
    parser.add_argument("--val-labels", required=True, help="CSV file containing validation labels")
    parser.add_argument("--val-meta", required=True, help="CSV file containing validation age and gender")
    parser.add_argument("--test-probs", required=True, help="CSV file containing test probabilities")
    parser.add_argument("--test-labels", required=True, help="CSV file containing test labels")
    parser.add_argument("--test-meta", required=True, help="CSV file containing test age and gender")
    parser.add_argument("--out", default="subgroup_thresholds_report.json", help="Output JSON report file")
    
    args = parser.parse_args()
    
    process_data(
        args.val_probs, args.val_labels, args.val_meta,
        args.test_probs, args.test_labels, args.test_meta,
        args.out
    )
