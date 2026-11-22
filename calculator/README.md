# TBI Mortality Risk Calculator

An English research interface for the frozen XGBoost Model A. It estimates death during the same hospitalization after the 24-hour ICU landmark for adults who meet the study cohort criteria.

This directory contains the source of the interactive research calculator. It runs with a FastAPI server. Cohort SHAP summaries are included. Participant-level data, fitted models, and reference datasets are kept in the study environment.

Enter the eight predictors on the left. Select **Predict mortality risk** to see the predicted probability and a case-specific SHAP waterfall on the right. The input order follows the selected cohort's published mean absolute SHAP ranking. The waterfall explains the submitted case in percentage points, from the mean training-reference prediction to the case prediction. When you submit another scenario in the same session, the page shows the difference from the previous successful submission.

## Run on Windows

1. Extract the contents of the ZIP into `D:/0mimic/TBI/06_SHAP_English_App/`.
2. In PowerShell, use the same Conda environment that ran the frozen model. These commands use the project's `jike` environment.

```powershell
Set-Location 'D:\0mimic\TBI\06_SHAP_English_App'
$conda = 'D:\miniconda\Scripts\conda.exe'
$modelEnv = 'D:\miniconda\envs\jike'

& $conda run -p $modelEnv --no-capture-output python -m pip install -r .\web_requirements.txt
if ($LASTEXITCODE -eq 0) {
    & $conda run -p $modelEnv --no-capture-output python .\serve_ranked.py --port 8001
}
```

3. The terminal should report that Model A and patient-level SHAP are ready. Keep it open and visit [the English calculator](http://127.0.0.1:8001/). Load the synthetic example, calculate risk, then change the GCS motor score and calculate again. Stop the server with `Ctrl+C`.

The default model is `D:/0mimic/TBI/04_Models/TBI_A_MI/XGBoost/model_bundle.joblib`. The SHAP reference is read from the original `D:/0mimic/TBI/04_Models/TBI_MI_Step1` folder. It uses `mimiciv_split_manifest.csv` and the `imp_XX/mimiciv_train.csv` files. Leave these files in the existing project folders. This ZIP contains only the app code and the published aggregate SHAP results.

If you stored these files elsewhere, pass `--bundle` and `--mi-dir` with their real paths. To update cohort-level rankings, pass `--rankings` with a compatible `results.json` from the same Model A analysis.

```powershell
& $conda run -p $modelEnv --no-capture-output python .\serve_ranked.py --port 8001 --bundle 'D:/path/model_bundle.joblib' --mi-dir 'D:/path/TBI_MI_Step1'
```

If the training reference cannot load, the app displays the predicted probability with a separate explanation status. Check the path reported in the PowerShell terminal and restart the service once the original training files are in place.

## Prediction and explanation

- The displayed number is the arithmetic mean of the frozen XGBoost members' mortality probabilities after each member's own fitted preprocessing. The app retains the original age cap, eight-variable set, eligibility criteria and complete-input requirement.
- Case-level SHAP enumerates all 256 subsets of the eight predictors over a fixed 100-stay MIMIC-IV training reference (seed 20261122) with the reference values from each completed training dataset. Selection is shared across imputations. For each subset, the app replaces the included training-reference values with the submitted values, then averages the resulting probabilities. It computes exact interventional Shapley values for this finite reference and averages them over the same ensemble members. Reference selection and background calculations take place on the local server.
- For each case, the app verifies that the baseline plus all eight contributions recovers the displayed probability within `1e-6` probability units. Contributions describe how the model arrived at the probability. They do not estimate how treatment or changing a measurement would affect the patient's outcome.
- The cohort selector affects the input order and global importance display. The patient waterfall uses the fixed training reference.
- Intended for research use. The app shows the probability without assigning a low/high risk category.

To compare the app's pipeline with the original validation predictions, run `python audit_parity.py --root 'D:/0mimic/TBI'` in the same environment. Load trusted study artifacts locally. For approved institutional access, run the app behind the institution's authenticated HTTPS reverse proxy on the same host, preserve the incoming Host header and set `--site-origin` to the real HTTPS origin. The server binds to the loopback interface.
