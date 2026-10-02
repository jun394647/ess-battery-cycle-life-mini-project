"""Rebuild the ESS analysis and reports in their required order.

Use --from-results to start from the versioned cell-level CSV files when the
large original MAT files have not been downloaded. That mode cannot verify
raw-file identity or recreate the Day 1 extraction/EDA figures.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from day1_eda import FILES

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def run(*args: str) -> None:
    command = [sys.executable, str(ROOT / 'src' / args[0]), *args[1:]]
    print('\n$ ' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-results', action='store_true',
                        help='Rebuild models and PDFs from versioned results without raw MAT files')
    args = parser.parse_args()

    if args.from_results:
        for name in ('all_cells.csv', 'paper_screened_cells.csv', 'batch_summary.csv',
                     'model_inputs/batch1.csv', 'model_inputs/batch2.csv', 'model_inputs/batch3.csv'):
            if not (RESULTS / name).is_file():
                parser.error(f'Missing results/{name}; run from the repository root with committed results')
    else:
        missing = [name for name in FILES.values() if not (ROOT / 'data' / name).is_file()]
        if missing:
            parser.error('Download the original MAT files listed in data/README.md first: '
                         + ', '.join(missing))
        for batch in FILES:
            run('day1_eda.py', batch)
        run('day1_compare.py')
        run('report_extra_figures.py')
        run('day1_model_probe.py')

    run('day2_model_compare.py', '--stage', 'screen')
    run('day2_feature_ablation.py')
    run('day2_refinement.py')
    run('day2_model.py', '--stage', 'select')
    run('day2_model.py', '--stage', 'evaluate', '--model', 'log_ridge_dq_qd_ir')
    run('day2_model_compare.py', '--stage', 'diagnose')
    run('day2_batch_calibration.py')
    run('day2_visuals.py')
    run('build_dashboard.py')
    run('audit_results.py', *(['--artifacts-only'] if args.from_results else []))
    run('build_day1_pdf.py')
    run('build_day1_pdf.py', '--day', '2')


if __name__ == '__main__':
    main()
