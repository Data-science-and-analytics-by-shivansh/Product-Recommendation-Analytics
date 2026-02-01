from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

def monitor_drift(reference_data, current_data):
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("drift_report.html")
    # Log to MLflow or alert if drift > threshold
