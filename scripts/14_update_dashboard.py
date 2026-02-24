"""
BlueprintLLM Dashboard Generator
================================
Parses training logs, exam results, and lesson files to generate
an interactive HTML dashboard showing model progress.

Usage:
    python scripts/14_update_dashboard.py

    Options:
        --training-log PATH    Path to training console output (default: auto-detect latest)
        --exam-dir PATH        Directory containing exam results (default: exams/)
        --lesson-dir PATH      Directory containing lesson files (default: lessons/)
        --model-dir PATH       Directory containing model configs (default: models/)
        --output PATH          Output HTML file (default: dashboard/index.html)
        --open                 Open in browser after generating

Examples:
    python scripts/14_update_dashboard.py
    python scripts/14_update_dashboard.py --open
    python scripts/14_update_dashboard.py --training-log logs/v2_training.log --open
"""

import argparse
import json
import os
import re
import sys
import glob
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_logger import get_logger as _get_pipeline_logger
plog = _get_pipeline_logger(step_prefix="9")


def load_health_report(root='.'):
    """Load the health report JSON if it exists."""
    path = os.path.join(root, 'health_report.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def parse_training_log(log_path):
    """Parse training console output for loss/accuracy metrics."""
    if not log_path or not os.path.exists(log_path):
        return None

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    steps = []
    evals = []
    metadata = {
        'model': 'unknown',
        'gpu': 'unknown',
        'epochs': 0,
        'dataset_size': 0,
        'lora_rank': 0,
        'seq_length': 0,
        'version': 'unknown',
    }

    # Extract metadata
    m = re.search(r'GPU:\s*(.+)', content)
    if m: metadata['gpu'] = m.group(1).strip()
    m = re.search(r'Model:\s*(.+)', content)
    if m: metadata['model'] = m.group(1).strip()
    m = re.search(r'Epochs:\s*(\d+)', content)
    if m: metadata['epochs'] = int(m.group(1))
    m = re.search(r'Dataset:\s*(\d+)', content)
    if m: metadata['dataset_size'] = int(m.group(1))
    m = re.search(r'LoRA rank:\s*(\d+)', content)
    if m: metadata['lora_rank'] = int(m.group(1))
    m = re.search(r'Max sequence length:\s*(\d+)', content)
    if m: metadata['seq_length'] = int(m.group(1))
    m = re.search(r'Output:\s*models/([\w-]+)', content)
    if m: metadata['version'] = m.group(1)

    # Parse training steps: {'loss': '0.4648', ... 'epoch': '0.3446'}
    pattern = r"\{'loss':\s*'([^']+)',\s*'grad_norm':\s*'[^']+',\s*'learning_rate':\s*'[^']+',\s*'entropy':\s*'[^']+',\s*'num_tokens':\s*'[^']+',\s*'mean_token_accuracy':\s*'([^']+)',\s*'epoch':\s*'([^']+)'\}"
    for m in re.finditer(pattern, content):
        steps.append({
            'epoch': float(m.group(3)),
            'loss': float(m.group(1)),
            'accuracy': float(m.group(2)),
        })

    # Parse eval steps
    eval_pattern = r"\{'eval_loss':\s*'([^']+)',.*?'eval_mean_token_accuracy':\s*'([^']+)',\s*'epoch':\s*'([^']+)'\}"
    for m in re.finditer(eval_pattern, content):
        evals.append({
            'epoch': float(m.group(3)),
            'loss': float(m.group(1)),
            'accuracy': float(m.group(2)),
        })

    # Also try the numeric format (no quotes around values)
    if not steps:
        pattern2 = r"\{'loss':\s*([0-9.e-]+),.*?'mean_token_accuracy':\s*([0-9.e-]+),\s*'epoch':\s*([0-9.e-]+)\}"
        for m in re.finditer(pattern2, content):
            steps.append({
                'epoch': float(m.group(3)),
                'loss': float(m.group(1)),
                'accuracy': float(m.group(2)),
            })

    if not evals:
        eval_pattern2 = r"\{'eval_loss':\s*([0-9.e-]+),.*?'eval_mean_token_accuracy':\s*([0-9.e-]+),\s*'epoch':\s*([0-9.e-]+)\}"
        for m in re.finditer(eval_pattern2, content):
            evals.append({
                'epoch': float(m.group(3)),
                'loss': float(m.group(1)),
                'accuracy': float(m.group(2)),
            })

    return {
        'metadata': metadata,
        'steps': steps,
        'evals': evals,
        'final_loss': steps[-1]['loss'] if steps else 0,
        'final_accuracy': steps[-1]['accuracy'] if steps else 0,
        'eval_accuracy': evals[-1]['accuracy'] if evals else 0,
        'initial_loss': steps[0]['loss'] if steps else 0,
        'total_steps': len(steps),
    }


def parse_exam_results(exam_dir):
    """Parse all exam result files and build node mastery map."""
    exams = []
    node_scores = {}  # node_name -> best score

    if not os.path.exists(exam_dir):
        return exams, node_scores

    # Find all exam summary files
    summaries = sorted(glob.glob(os.path.join(exam_dir, 'exam_*_summary.json')))
    details = sorted(glob.glob(os.path.join(exam_dir, 'exam_*.jsonl')))

    for summary_path in summaries:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        exams.append(summary)

    # Parse detailed results for node-level scores
    for detail_path in details:
        if detail_path.endswith('_summary.json'):
            continue
        with open(detail_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract node types from the expected DSL
                prompt_id = entry.get('prompt_id', '')
                score = entry.get('score', 0)
                
                # Try to identify which nodes this prompt tests
                expected = entry.get('expected_dsl', '')
                nodes_in_example = set()
                for line_text in expected.split('\n'):
                    m = re.match(r'NODE\s+\w+:\s*(\w+)', line_text)
                    if m:
                        nodes_in_example.add(m.group(1))

                for node_name in nodes_in_example:
                    if node_name not in node_scores or score > node_scores[node_name]:
                        node_scores[node_name] = score

    return exams, node_scores


def parse_lessons(lesson_dir):
    """Parse lesson files for timeline display."""
    lessons = []
    if not os.path.exists(lesson_dir):
        return lessons

    for path in sorted(glob.glob(os.path.join(lesson_dir, 'lesson_*.json'))):
        with open(path, 'r') as f:
            data = json.load(f)
        
        lesson_num = re.search(r'lesson_(\d+)', path)
        num = int(lesson_num.group(1)) if lesson_num else 0
        
        lessons.append({
            'id': num,
            'title': data.get('title', f'Lesson {num:02d}'),
            'prompts': len(data.get('prompts', [])),
            'description': data.get('description', ''),
        })

    return lessons


def detect_latest_training_log(models_dir, logs_dir=None):
    """Auto-detect the most recent training log."""
    candidates = []
    
    # Check for training logs in various locations
    search_paths = [
        'logs/*.log',
        'logs/*.txt',
        'training_*.log',
        'training_*.txt',
        'models/*/training.log',
    ]
    if logs_dir:
        search_paths.insert(0, os.path.join(logs_dir, '*.log'))
        search_paths.insert(0, os.path.join(logs_dir, '*.txt'))

    for pattern in search_paths:
        candidates.extend(glob.glob(pattern))

    if not candidates:
        return None

    # Return most recently modified
    return max(candidates, key=os.path.getmtime)


def build_node_mastery_list(node_scores):
    """Build the full node mastery list with scores from exams."""
    all_nodes = [
        ("BeginPlay", "Events"), ("PrintString", "Debug"), ("Delay", "Flow"),
        ("Branch", "Flow"), ("Sequence", "Flow"), ("FlipFlop", "Flow"),
        ("DoOnce", "Flow"), ("Gate", "Flow"), ("ForEachLoop", "Flow"),
        ("WhileLoop", "Flow"), ("CustomEvent", "Events"), ("EventTick", "Events"),
        ("InputAction", "Input"), ("OnOverlap", "Collision"), ("OnHit", "Collision"),
        ("SpawnActor", "Actor"), ("DestroyActor", "Actor"),
        ("SetActorLocation", "Transform"), ("GetActorLocation", "Transform"),
        ("AddActorOffset", "Transform"), ("SetActorRotation", "Transform"),
        ("SetVisibility", "Rendering"), ("SetStaticMesh", "Rendering"),
        ("PlaySound", "Audio"), ("PlayAnimMontage", "Animation"),
        ("Timeline", "Animation"), ("SetTimer", "Timer"), ("ClearTimer", "Timer"),
        ("CastTo", "Casting"), ("IsValid", "Utility"),
        ("GetPlayerCharacter", "Player"), ("GetPlayerController", "Player"),
        ("SetVariable", "Variables"), ("GetVariable", "Variables"),
        ("MathAdd", "Math"), ("MathMultiply", "Math"),
        ("VectorLength", "Math"), ("GetDistanceTo", "Math"), ("Lerp", "Math"),
        ("MakeArray", "Arrays"), ("ArrayAdd", "Arrays"), ("ArrayGet", "Arrays"),
    ]

    result = []
    for name, category in all_nodes:
        score = node_scores.get(name, None)
        # Also check common aliases
        if score is None:
            for key, val in node_scores.items():
                if key.lower() == name.lower() or name.lower() in key.lower():
                    score = val
                    break
        result.append({'name': name, 'category': category, 'score': score})

    # Add any nodes from exams not in the default list
    known = {n.lower() for n, _ in all_nodes}
    for name, score in node_scores.items():
        if name.lower() not in known:
            result.append({'name': name, 'category': 'Discovered', 'score': score})

    return result


def build_activity_log(training_data, exams, lessons_on_disk):
    """Build recent activity entries."""
    logs = []
    now = datetime.now()

    if training_data and training_data['steps']:
        loss = training_data['final_loss']
        acc = training_data['final_accuracy']
        ver = training_data['metadata'].get('version', '??')
        logs.append({
            'time': now.strftime('%I:%M %p'),
            'msg': f'{ver} training complete — loss: {loss:.3f}, accuracy: {acc*100:.1f}%',
            'level': 'success',
        })
        initial = training_data['initial_loss']
        logs.append({
            'time': '',
            'msg': f'Loss reduced from {initial:.3f} → {loss:.3f} ({(1 - loss/initial)*100:.0f}% reduction)',
            'level': 'info',
        })

    for exam in reversed(exams[-3:]):
        score = exam.get('overall_score', exam.get('average_score', 0))
        lesson = exam.get('lesson', '??')
        logs.append({
            'time': '',
            'msg': f'Exam {lesson}: {score*100:.0f}% score',
            'level': 'success' if score > 0.8 else 'warning' if score > 0.5 else 'error',
        })

    if not logs:
        logs.append({'time': now.strftime('%I:%M %p'), 'msg': 'Dashboard initialized — awaiting first training run', 'level': 'info'})

    return logs


def generate_dashboard_html(training_data, node_mastery, lessons, exams, activity_log, error_breakdown, health_report=None):
    """Generate the complete dashboard HTML with embedded data."""
    
    # Prepare JSON data for embedding
    loss_history_json = json.dumps(training_data['steps'] if training_data else [])
    eval_points_json = json.dumps(training_data['evals'] if training_data else [])
    node_mastery_json = json.dumps(node_mastery)
    
    # Compute summary stats
    if training_data and training_data['steps']:
        final_loss = training_data['final_loss']
        final_acc = training_data['final_accuracy']
        eval_acc = training_data['eval_accuracy']
        initial_loss = training_data['initial_loss']
        loss_reduction = (1 - final_loss / initial_loss) * 100 if initial_loss > 0 else 0
        total_steps = training_data['total_steps']
        max_epoch = training_data['steps'][-1]['epoch']
        version = training_data['metadata'].get('version', 'v?')
        model_size = '8B' if '8b' in training_data['metadata'].get('model', '').lower() or '8B' in training_data['metadata'].get('model', '') else '3B'
        gpu = training_data['metadata'].get('gpu', 'Unknown GPU')
        status = 'idle'
    else:
        final_loss = 0
        final_acc = 0
        eval_acc = 0
        initial_loss = 0
        loss_reduction = 0
        total_steps = 0
        max_epoch = 0
        version = 'v0'
        model_size = '?'
        gpu = 'Unknown'
        status = 'idle'

    mastered = sum(1 for n in node_mastery if n['score'] is not None and n['score'] >= 0.85)
    learning = sum(1 for n in node_mastery if n['score'] is not None and 0.6 <= n['score'] < 0.85)
    struggling = sum(1 for n in node_mastery if n['score'] is not None and n['score'] < 0.6)
    total_nodes = len(node_mastery)
    completed_cycles = sum(1 for e in exams)

    # Build lessons data for timeline
    lessons_data = []
    exam_lesson_ids = set()
    for exam in exams:
        lid = exam.get('lesson_id', exam.get('lesson', 0))
        exam_lesson_ids.add(lid)

    for lesson in lessons:
        lid = lesson['id']
        has_exam = lid in exam_lesson_ids
        exam_score = None
        for exam in exams:
            if exam.get('lesson_id', exam.get('lesson', 0)) == lid:
                exam_score = exam.get('overall_score', exam.get('average_score', 0))

        if has_exam and exam_score is not None:
            s = 'complete'
        elif lid == (max(exam_lesson_ids) + 1 if exam_lesson_ids else 1):
            s = 'active'
        else:
            s = 'pending'
        
        lessons_data.append({
            'id': lid,
            'title': lesson.get('title', f'Lesson {lid:02d}'),
            'prompts': lesson.get('prompts', 0),
            'status': s,
            'score': round(exam_score * 100) if exam_score is not None else None,
        })

    # Ensure at least 5 lessons shown
    existing_ids = {l['id'] for l in lessons_data}
    for i in range(1, 6):
        if i not in existing_ids:
            titles = {
                1: 'Lesson 01 — Core Patterns',
                2: 'Lesson 02 — Complex Flow',
                3: 'Lesson 03 — Actor Systems',
                4: 'Lesson 04 — Animation & Audio',
                5: 'Lesson 05 — Advanced Patterns',
            }
            lessons_data.append({
                'id': i,
                'title': titles.get(i, f'Lesson {i:02d}'),
                'prompts': 0,
                'status': 'active' if i == 1 and not exam_lesson_ids else 'pending',
                'score': None,
            })
    lessons_data.sort(key=lambda x: x['id'])
    lessons_json = json.dumps(lessons_data[:5])

    # Activity log
    log_html = ''
    for entry in activity_log[:8]:
        log_html += f'''<div class="log-line">
            <span class="log-time">{entry["time"]}</span>
            <span class="log-msg {entry["level"]}">{entry["msg"]}</span>
        </div>\n'''

    # Error breakdown
    err = error_breakdown or {}
    err_missing_nodes = err.get('missing_nodes', 0)
    err_missing_exec = err.get('missing_exec', 0)
    err_missing_data = err.get('missing_data', 0)
    err_extra = err.get('extra_lines', 0)
    err_format = err.get('format_errors', 0)
    err_max = max(err_missing_nodes, err_missing_exec, err_missing_data, err_extra, err_format, 1)

    # Read the HTML template and inject data
    html = DASHBOARD_TEMPLATE
    
    # Replace placeholders
    html = html.replace('{{VERSION}}', version)
    html = html.replace('{{MODEL_SIZE}}', model_size)
    html = html.replace('{{GPU}}', gpu)
    html = html.replace('{{STATUS}}', status)
    html = html.replace('{{FINAL_LOSS}}', f'{final_loss:.3f}')
    html = html.replace('{{INITIAL_LOSS}}', f'{initial_loss:.3f}')
    html = html.replace('{{LOSS_REDUCTION}}', f'{loss_reduction:.0f}')
    html = html.replace('{{FINAL_ACCURACY}}', f'{final_acc*100:.1f}')
    html = html.replace('{{EVAL_ACCURACY}}', f'{eval_acc*100:.1f}')
    html = html.replace('{{NODES_MASTERED}}', str(mastered))
    html = html.replace('{{NODES_TOTAL}}', str(total_nodes))
    html = html.replace('{{NODES_LEARNING}}', str(learning))
    html = html.replace('{{NODES_STRUGGLING}}', str(struggling))
    html = html.replace('{{COMPLETED_CYCLES}}', str(completed_cycles))
    html = html.replace('{{TOTAL_STEPS}}', str(total_steps))
    html = html.replace('{{MAX_EPOCH}}', f'{max_epoch:.1f}')
    html = html.replace('{{LOSS_HISTORY_JSON}}', loss_history_json)
    html = html.replace('{{EVAL_POINTS_JSON}}', eval_points_json)
    html = html.replace('{{NODE_MASTERY_JSON}}', node_mastery_json)
    html = html.replace('{{LESSONS_JSON}}', lessons_json)
    html = html.replace('{{LOG_HTML}}', log_html)
    html = html.replace('{{ERR_MISSING_NODES}}', str(err_missing_nodes))
    html = html.replace('{{ERR_MISSING_EXEC}}', str(err_missing_exec))
    html = html.replace('{{ERR_MISSING_DATA}}', str(err_missing_data))
    html = html.replace('{{ERR_EXTRA}}', str(err_extra))
    html = html.replace('{{ERR_FORMAT}}', str(err_format))
    html = html.replace('{{ERR_MAX}}', str(err_max))
    html = html.replace('{{GENERATED_TIME}}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Fix error bar values in the template
    def err_pct(val):
        return str(round((val / err_max) * 100)) if err_max > 0 and val > 0 else '0'

    html = html.replace('ERRPCT_NODES', err_pct(err_missing_nodes))
    html = html.replace('ERRPCT_EXEC', err_pct(err_missing_exec))
    html = html.replace('ERRPCT_DATA', err_pct(err_missing_data))
    html = html.replace('ERRPCT_EXTRA', err_pct(err_extra))
    html = html.replace('ERRPCT_FORMAT', err_pct(err_format))
    html = html.replace('ERRVAL_NODES', str(err_missing_nodes) if err_missing_nodes > 0 else '—')
    html = html.replace('ERRVAL_EXEC', str(err_missing_exec) if err_missing_exec > 0 else '—')
    html = html.replace('ERRVAL_DATA', str(err_missing_data) if err_missing_data > 0 else '—')
    html = html.replace('ERRVAL_EXTRA', str(err_extra) if err_extra > 0 else '—')
    html = html.replace('ERRVAL_FORMAT', str(err_format) if err_format > 0 else '—')

    # Health report data
    if health_report:
        summary = health_report.get('summary', {})
        by_level = summary.get('by_level', {})
        overall = summary.get('overall_health', 'unknown')
        health_badge_class = {
            'healthy': 'health-ok', 'suggestion': 'health-ok',
            'warning': 'health-warn', 'critical': 'health-crit',
        }.get(overall, 'health-ok')
        html = html.replace('{{HEALTH_BADGE_CLASS}}', health_badge_class)
        html = html.replace('{{HEALTH_OVERALL}}', overall.upper())
        html = html.replace('{{HEALTH_TOTAL}}', str(summary.get('total_alerts', 0)))
        html = html.replace('{{HEALTH_CRITICAL}}', str(by_level.get('CRITICAL', 0)))
        html = html.replace('{{HEALTH_WARNING}}', str(by_level.get('WARNING', 0)))
        html = html.replace('{{HEALTH_SUGGESTION}}', str(by_level.get('SUGGESTION', 0)))
        html = html.replace('{{HEALTH_INFO}}', str(by_level.get('INFO', 0)))
        html = html.replace('{{HEALTH_ALERTS_JSON}}', json.dumps(health_report.get('alerts', [])))
        html = html.replace('{{HEALTH_SUMMARY_JSON}}', json.dumps(summary))
        html = html.replace('{{HEALTH_DISPLAY}}', '')
    else:
        html = html.replace('{{HEALTH_BADGE_CLASS}}', 'health-ok')
        html = html.replace('{{HEALTH_OVERALL}}', 'NO DATA')
        html = html.replace('{{HEALTH_TOTAL}}', '0')
        html = html.replace('{{HEALTH_CRITICAL}}', '0')
        html = html.replace('{{HEALTH_WARNING}}', '0')
        html = html.replace('{{HEALTH_SUGGESTION}}', '0')
        html = html.replace('{{HEALTH_INFO}}', '0')
        html = html.replace('{{HEALTH_ALERTS_JSON}}', '[]')
        html = html.replace('{{HEALTH_SUMMARY_JSON}}', '{}')
        html = html.replace('{{HEALTH_DISPLAY}}', ' style="opacity:0.5"')

    return html


# ═══════════════════════════════════════════════════
# HTML TEMPLATE
# ═══════════════════════════════════════════════════

DASHBOARD_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BlueprintLLM — Training Observatory</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg-deep: #0a0e17;
  --bg-panel: #111827;
  --bg-card: #1a2234;
  --bg-card-hover: #1f2a3f;
  --border: #2a3650;
  --border-bright: #3b4f6e;
  --text-primary: #e8edf5;
  --text-secondary: #8899b4;
  --text-dim: #556680;
  --accent-blue: #3b82f6;
  --accent-cyan: #06b6d4;
  --accent-emerald: #10b981;
  --accent-amber: #f59e0b;
  --accent-rose: #f43f5e;
  --accent-violet: #8b5cf6;
  --glow-blue: rgba(59, 130, 246, 0.15);
  --glow-emerald: rgba(16, 185, 129, 0.15);
  --glow-amber: rgba(245, 158, 11, 0.15);
  --glow-rose: rgba(244, 63, 94, 0.15);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: var(--bg-deep); color: var(--text-primary); font-family: 'Outfit', sans-serif; min-height: 100vh; overflow-x: hidden; }
body::before { content: ''; position: fixed; inset: 0; background-image: linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px); background-size: 60px 60px; pointer-events: none; z-index: 0; }
body::after { content: ''; position: fixed; top: -200px; left: 50%; transform: translateX(-50%); width: 800px; height: 500px; background: radial-gradient(ellipse, rgba(59,130,246,0.08) 0%, transparent 70%); pointer-events: none; z-index: 0; }
.dashboard { position: relative; z-index: 1; max-width: 1440px; margin: 0 auto; padding: 32px 40px; }
.header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 36px; padding-bottom: 24px; border-bottom: 1px solid var(--border); }
.header-left { display: flex; align-items: center; gap: 16px; }
.logo { width: 44px; height: 44px; background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 18px; color: white; box-shadow: 0 4px 20px rgba(59,130,246,0.3); }
.header-title { font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }
.header-title span { color: var(--text-dim); font-weight: 400; }
.header-right { display: flex; align-items: center; gap: 20px; }
.status-pill { display: flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 500; font-family: 'JetBrains Mono', monospace; }
.status-pill.training { background: var(--glow-amber); color: var(--accent-amber); border: 1px solid rgba(245,158,11,0.2); }
.status-pill.idle { background: var(--glow-emerald); color: var(--accent-emerald); border: 1px solid rgba(16,185,129,0.2); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; animation: pulse 2s ease-in-out infinite; }
.training .status-dot { background: var(--accent-amber); }
.idle .status-dot { background: var(--accent-emerald); }
@keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.5; transform:scale(0.85); } }
.version-badge { padding: 6px 12px; border-radius: 8px; background: var(--bg-card); border: 1px solid var(--border); font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-secondary); }
.generated-time { font-size: 11px; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }
.metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }
.metric-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; padding: 20px 22px; position: relative; overflow: hidden; transition: border-color 0.3s, transform 0.2s; }
.metric-card:hover { border-color: var(--border-bright); transform: translateY(-2px); }
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.metric-card.blue::before { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
.metric-card.emerald::before { background: linear-gradient(90deg, var(--accent-emerald), var(--accent-cyan)); }
.metric-card.amber::before { background: linear-gradient(90deg, var(--accent-amber), #fbbf24); }
.metric-card.violet::before { background: linear-gradient(90deg, var(--accent-violet), var(--accent-blue)); }
.metric-label { font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim); margin-bottom: 8px; }
.metric-value { font-size: 36px; font-weight: 800; letter-spacing: -1px; line-height: 1; margin-bottom: 6px; }
.metric-card.blue .metric-value { color: var(--accent-blue); }
.metric-card.emerald .metric-value { color: var(--accent-emerald); }
.metric-card.amber .metric-value { color: var(--accent-amber); }
.metric-card.violet .metric-value { color: var(--accent-violet); }
.metric-sub { font-size: 13px; color: var(--text-secondary); font-family: 'JetBrains Mono', monospace; }
.metric-trend { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 6px; margin-left: 8px; }
.metric-trend.up { background: var(--glow-emerald); color: var(--accent-emerald); }
.main-grid { display: grid; grid-template-columns: 1fr 380px; gap: 20px; margin-bottom: 28px; }
.panel { background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; padding: 22px; }
.panel-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; }
.panel-title { font-size: 15px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.panel-title .icon { width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-size: 14px; }
.panel-badge { font-size: 11px; font-family: 'JetBrains Mono', monospace; padding: 3px 10px; border-radius: 6px; background: var(--bg-deep); color: var(--text-dim); border: 1px solid var(--border); }
.chart-container { position: relative; height: 220px; margin-top: 8px; }
svg text { font-family: 'JetBrains Mono', monospace; }
.chart-line { fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.chart-area { opacity: 0.1; }
.chart-grid line { stroke: var(--border); stroke-dasharray: 4 4; stroke-width: 0.5; }
.node-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; max-height: 420px; overflow-y: auto; scrollbar-width: thin; scrollbar-color: var(--border) transparent; }
.node-item { display: flex; align-items: center; gap: 10px; padding: 10px 12px; background: var(--bg-deep); border-radius: 8px; border: 1px solid transparent; transition: border-color 0.2s, background 0.2s; }
.node-item:hover { background: var(--bg-card-hover); border-color: var(--border); }
.node-status { width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; }
.node-status.mastered { background: var(--accent-emerald); box-shadow: 0 0 8px rgba(16,185,129,0.4); }
.node-status.learning { background: var(--accent-amber); box-shadow: 0 0 8px rgba(245,158,11,0.3); }
.node-status.untrained { background: var(--border); }
.node-status.struggling { background: var(--accent-rose); box-shadow: 0 0 8px rgba(244,63,94,0.3); }
.node-name { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 500; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.node-score { font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; min-width: 36px; text-align: right; }
.node-score.high { color: var(--accent-emerald); }
.node-score.mid { color: var(--accent-amber); }
.node-score.low { color: var(--accent-rose); }
.node-score.none { color: var(--text-dim); }
.timeline { display: flex; flex-direction: column; gap: 2px; }
.timeline-item { display: flex; align-items: stretch; gap: 14px; padding: 12px 0; }
.timeline-line { display: flex; flex-direction: column; align-items: center; width: 20px; flex-shrink: 0; }
.timeline-dot { width: 12px; height: 12px; border-radius: 50%; border: 2px solid var(--border); flex-shrink: 0; }
.timeline-dot.complete { background: var(--accent-emerald); border-color: var(--accent-emerald); }
.timeline-dot.active { background: var(--accent-blue); border-color: var(--accent-blue); box-shadow: 0 0 12px rgba(59,130,246,0.5); }
.timeline-dot.pending { background: transparent; border-color: var(--border); }
.timeline-connector { width: 2px; flex: 1; background: var(--border); margin: 4px 0; }
.timeline-content { flex: 1; padding-bottom: 8px; }
.timeline-title { font-size: 14px; font-weight: 600; margin-bottom: 4px; }
.timeline-meta { font-size: 12px; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }
.timeline-score { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; font-family: 'JetBrains Mono', monospace; margin-top: 6px; }
.timeline-score.good { background: var(--glow-emerald); color: var(--accent-emerald); }
.timeline-score.ok { background: var(--glow-amber); color: var(--accent-amber); }
.timeline-score.pending-score { background: var(--bg-deep); color: var(--text-dim); }
.bottom-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.log-container { max-height: 200px; overflow-y: auto; scrollbar-width: thin; scrollbar-color: var(--border) transparent; }
.log-line { display: flex; align-items: baseline; gap: 10px; padding: 4px 0; font-family: 'JetBrains Mono', monospace; font-size: 12px; line-height: 1.6; border-bottom: 1px solid rgba(42,54,80,0.3); }
.log-time { color: var(--text-dim); flex-shrink: 0; font-size: 11px; min-width: 60px; }
.log-msg { color: var(--text-secondary); }
.log-msg.success { color: var(--accent-emerald); }
.log-msg.warning { color: var(--accent-amber); }
.log-msg.error { color: var(--accent-rose); }
.log-msg.info { color: var(--accent-cyan); }
.error-bars { display: flex; flex-direction: column; gap: 12px; margin-top: 4px; }
.error-bar-row { display: flex; align-items: center; gap: 12px; }
.error-bar-label { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-secondary); width: 120px; flex-shrink: 0; }
.error-bar-track { flex: 1; height: 8px; background: var(--bg-deep); border-radius: 4px; overflow: hidden; }
.error-bar-fill { height: 100%; border-radius: 4px; transition: width 0.8s ease; }
.error-bar-fill.rose { background: linear-gradient(90deg, var(--accent-rose), #fb7185); }
.error-bar-fill.amber { background: linear-gradient(90deg, var(--accent-amber), #fbbf24); }
.error-bar-fill.blue { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
.error-bar-fill.violet { background: linear-gradient(90deg, var(--accent-violet), #a78bfa); }
.error-bar-value { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; color: var(--text-secondary); width: 36px; text-align: right; }
.legend { display: flex; gap: 20px; margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-secondary); }
.legend-dot { width: 8px; height: 8px; border-radius: 3px; }
@media (max-width: 1100px) { .main-grid { grid-template-columns: 1fr; } .bottom-grid { grid-template-columns: 1fr; } .metrics-row { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 640px) { .dashboard { padding: 16px; } .metrics-row { grid-template-columns: 1fr; } .node-grid { grid-template-columns: 1fr; } .header { flex-direction: column; gap: 12px; } }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-bright); }
.health-grid { display: grid; grid-template-columns: 200px 1fr; gap: 20px; margin-top: 28px; }
.health-status { display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 20px; background: var(--bg-deep); border-radius: 12px; }
.health-badge { font-size: 15px; font-weight: 700; font-family: 'JetBrains Mono', monospace; padding: 8px 18px; border-radius: 8px; text-align: center; }
.health-badge.health-ok { background: var(--glow-emerald); color: var(--accent-emerald); border: 1px solid rgba(16,185,129,0.3); }
.health-badge.health-warn { background: var(--glow-amber); color: var(--accent-amber); border: 1px solid rgba(245,158,11,0.3); }
.health-badge.health-crit { background: var(--glow-rose); color: var(--accent-rose); border: 1px solid rgba(244,63,94,0.3); }
.health-counts { display: flex; flex-direction: column; gap: 6px; width: 100%; }
.health-count { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 12px; padding: 4px 8px; border-radius: 4px; }
.health-count.crit { color: var(--accent-rose); background: rgba(244,63,94,0.08); }
.health-count.warn { color: var(--accent-amber); background: rgba(245,158,11,0.08); }
.health-count.sugg { color: var(--accent-blue); background: rgba(59,130,246,0.08); }
.health-count.info { color: var(--accent-cyan); background: rgba(6,182,212,0.08); }
.health-alert-list { max-height: 260px; overflow-y: auto; scrollbar-width: thin; scrollbar-color: var(--border) transparent; display: flex; flex-direction: column; gap: 6px; }
.health-alert-item { padding: 10px 14px; background: var(--bg-deep); border-radius: 8px; border-left: 3px solid var(--border); }
.health-alert-item.alert-CRITICAL { border-left-color: var(--accent-rose); }
.health-alert-item.alert-WARNING { border-left-color: var(--accent-amber); }
.health-alert-item.alert-SUGGESTION { border-left-color: var(--accent-blue); }
.health-alert-item.alert-INFO { border-left-color: var(--accent-cyan); }
.health-alert-title { font-size: 13px; font-weight: 600; margin-bottom: 4px; display: flex; align-items: center; gap: 8px; }
.health-alert-title .alert-tag { font-size: 10px; font-family: 'JetBrains Mono', monospace; padding: 1px 6px; border-radius: 3px; font-weight: 500; }
.alert-tag.tag-CRITICAL { background: var(--glow-rose); color: var(--accent-rose); }
.alert-tag.tag-WARNING { background: var(--glow-amber); color: var(--accent-amber); }
.alert-tag.tag-SUGGESTION { background: rgba(59,130,246,0.15); color: var(--accent-blue); }
.alert-tag.tag-INFO { background: rgba(6,182,212,0.15); color: var(--accent-cyan); }
.health-alert-detail { font-size: 12px; color: var(--text-secondary); line-height: 1.5; }
@media (max-width: 900px) { .health-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="dashboard">
  <header class="header">
    <div class="header-left">
      <div class="logo">B</div>
      <div class="header-title">BlueprintLLM <span>Training Observatory</span></div>
    </div>
    <div class="header-right">
      <div class="generated-time">Updated {{GENERATED_TIME}}</div>
      <div class="status-pill idle">
        <div class="status-dot"></div>
        <span>{{VERSION}}</span>
      </div>
      <div class="version-badge">{{VERSION}} · {{MODEL_SIZE}} · {{GPU}}</div>
    </div>
  </header>
  <div class="metrics-row">
    <div class="metric-card blue">
      <div class="metric-label">Training Loss</div>
      <div class="metric-value">{{FINAL_LOSS}}</div>
      <div class="metric-sub">from {{INITIAL_LOSS}} <span class="metric-trend up">↓ {{LOSS_REDUCTION}}%</span></div>
    </div>
    <div class="metric-card emerald">
      <div class="metric-label">Token Accuracy</div>
      <div class="metric-value">{{FINAL_ACCURACY}}%</div>
      <div class="metric-sub">eval: {{EVAL_ACCURACY}}% <span class="metric-trend up">↑ generalizing</span></div>
    </div>
    <div class="metric-card amber">
      <div class="metric-label">Nodes Mastered</div>
      <div class="metric-value">{{NODES_MASTERED}}</div>
      <div class="metric-sub">of {{NODES_TOTAL}} target nodes</div>
    </div>
    <div class="metric-card violet">
      <div class="metric-label">Teaching Cycles</div>
      <div class="metric-value">{{COMPLETED_CYCLES}}</div>
      <div class="metric-sub">{{NODES_LEARNING}} learning · {{NODES_STRUGGLING}} struggling</div>
    </div>
  </div>
  <div class="main-grid">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title"><span class="icon">📉</span> Training Loss Curve</div>
        <div class="panel-badge">{{MAX_EPOCH}} epochs · {{TOTAL_STEPS}} steps</div>
      </div>
      <div class="chart-container">
        <svg id="lossChart" viewBox="0 0 700 200" preserveAspectRatio="none" style="width:100%;height:100%">
          <g class="chart-grid">
            <line x1="40" y1="10" x2="690" y2="10"/>
            <line x1="40" y1="57" x2="690" y2="57"/>
            <line x1="40" y1="105" x2="690" y2="105"/>
            <line x1="40" y1="152" x2="690" y2="152"/>
            <line x1="40" y1="180" x2="690" y2="180"/>
          </g>
          <text x="35" y="14" text-anchor="end" fill="#556680" font-size="9">2.5</text>
          <text x="35" y="61" text-anchor="end" fill="#556680" font-size="9">1.8</text>
          <text x="35" y="109" text-anchor="end" fill="#556680" font-size="9">1.0</text>
          <text x="35" y="156" text-anchor="end" fill="#556680" font-size="9">0.5</text>
          <text x="35" y="184" text-anchor="end" fill="#556680" font-size="9">0.2</text>
          <text x="40" y="197" text-anchor="middle" fill="#556680" font-size="9">0</text>
          <text x="365" y="197" text-anchor="middle" fill="#556680" font-size="9">1.0</text>
          <text x="690" y="197" text-anchor="middle" fill="#556680" font-size="9">2.0</text>
          <path class="chart-area" fill="url(#blueGrad)" d=""/>
          <path id="lossLine" class="chart-line" stroke="var(--accent-blue)" d=""/>
          <g id="evalDots"></g>
          <defs>
            <linearGradient id="blueGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="var(--accent-blue)" stop-opacity="0.3"/>
              <stop offset="100%" stop-color="var(--accent-blue)" stop-opacity="0"/>
            </linearGradient>
          </defs>
        </svg>
      </div>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:var(--accent-blue)"></div> Train Loss</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--accent-cyan)"></div> Eval Loss</div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title"><span class="icon">🧠</span> Node Mastery</div>
        <div class="panel-badge" id="masteryBadge">{{NODES_MASTERED}} mastered · {{NODES_LEARNING}} learning</div>
      </div>
      <div class="node-grid" id="nodeGrid"></div>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:var(--accent-emerald)"></div> Mastered</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--accent-amber)"></div> Learning</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--accent-rose)"></div> Struggling</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--border)"></div> Untrained</div>
      </div>
    </div>
  </div>
  <div class="bottom-grid">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title"><span class="icon">🔄</span> Teaching Loop Progress</div>
        <div class="panel-badge">cycle {{COMPLETED_CYCLES}}</div>
      </div>
      <div class="timeline" id="timeline"></div>
    </div>
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title"><span class="icon">🔍</span> Error Categories</div>
        <div class="panel-badge">from last exam</div>
      </div>
      <div class="error-bars">
        <div class="error-bar-row">
          <div class="error-bar-label">Missing Nodes</div>
          <div class="error-bar-track"><div class="error-bar-fill rose" style="width:ERRPCT_NODES%"></div></div>
          <div class="error-bar-value">ERRVAL_NODES</div>
        </div>
        <div class="error-bar-row">
          <div class="error-bar-label">Missing EXEC</div>
          <div class="error-bar-track"><div class="error-bar-fill amber" style="width:ERRPCT_EXEC%"></div></div>
          <div class="error-bar-value">ERRVAL_EXEC</div>
        </div>
        <div class="error-bar-row">
          <div class="error-bar-label">Missing DATA</div>
          <div class="error-bar-track"><div class="error-bar-fill blue" style="width:ERRPCT_DATA%"></div></div>
          <div class="error-bar-value">ERRVAL_DATA</div>
        </div>
        <div class="error-bar-row">
          <div class="error-bar-label">Extra Lines</div>
          <div class="error-bar-track"><div class="error-bar-fill violet" style="width:ERRPCT_EXTRA%"></div></div>
          <div class="error-bar-value">ERRVAL_EXTRA</div>
        </div>
        <div class="error-bar-row">
          <div class="error-bar-label">Format Errors</div>
          <div class="error-bar-track"><div class="error-bar-fill rose" style="width:ERRPCT_FORMAT%"></div></div>
          <div class="error-bar-value">ERRVAL_FORMAT</div>
        </div>
      </div>
      <div style="margin-top:20px; padding-top:16px; border-top:1px solid var(--border)">
        <div class="panel-title" style="margin-bottom:10px; font-size:13px;">
          <span class="icon">📋</span> Recent Activity
        </div>
        <div class="log-container">{{LOG_HTML}}</div>
      </div>
    </div>
  </div>
  <div class="panel health-grid-panel" style="margin-top:28px"{{HEALTH_DISPLAY}}>
    <div class="panel-header">
      <div class="panel-title"><span class="icon">🩺</span> Training Health Monitor</div>
      <div class="panel-badge">{{HEALTH_TOTAL}} alerts</div>
    </div>
    <div class="health-grid">
      <div class="health-status">
        <div class="health-badge {{HEALTH_BADGE_CLASS}}">{{HEALTH_OVERALL}}</div>
        <div class="health-counts">
          <div class="health-count crit"><span>Critical</span><span>{{HEALTH_CRITICAL}}</span></div>
          <div class="health-count warn"><span>Warning</span><span>{{HEALTH_WARNING}}</span></div>
          <div class="health-count sugg"><span>Suggestion</span><span>{{HEALTH_SUGGESTION}}</span></div>
          <div class="health-count info"><span>Info</span><span>{{HEALTH_INFO}}</span></div>
        </div>
      </div>
      <div class="health-alert-list" id="healthAlerts"></div>
    </div>
  </div>
</div>
<script>
const LOSS_HISTORY = {{LOSS_HISTORY_JSON}};
const EVAL_POINTS = {{EVAL_POINTS_JSON}};
const NODE_MASTERY = {{NODE_MASTERY_JSON}};
const LESSONS = {{LESSONS_JSON}};
const HEALTH_ALERTS = {{HEALTH_ALERTS_JSON}};
const HEALTH_SUMMARY = {{HEALTH_SUMMARY_JSON}};

function renderChart() {
  if (!LOSS_HISTORY.length) return;
  const maxEpoch = Math.max(...LOSS_HISTORY.map(p => p.epoch), 2);
  const maxLoss = 2.6;
  const cL=44, cR=690, cT=8, cB=182, cW=cR-cL, cH=cB-cT;
  const x = e => cL + (e/maxEpoch)*cW;
  const y = l => cB - (l/maxLoss)*cH;
  let d = LOSS_HISTORY.map((p,i) => `${i===0?'M':'L'}${x(p.epoch)},${y(p.loss)}`).join(' ');
  document.getElementById('lossLine').setAttribute('d', d);
  const last = LOSS_HISTORY[LOSS_HISTORY.length-1], first = LOSS_HISTORY[0];
  document.querySelector('.chart-area').setAttribute('d', d + ` L${x(last.epoch)},${cB} L${x(first.epoch)},${cB} Z`);
  const eg = document.getElementById('evalDots');
  eg.innerHTML = EVAL_POINTS.map(p => `<circle cx="${x(p.epoch)}" cy="${y(p.loss)}" r="5" fill="var(--accent-cyan)" stroke="var(--bg-deep)" stroke-width="2"/><text x="${x(p.epoch)}" y="${y(p.loss)-10}" text-anchor="middle" fill="var(--accent-cyan)" font-size="9">${p.loss.toFixed(3)}</text>`).join('');
}

function renderNodes() {
  const g = document.getElementById('nodeGrid');
  const sorted = [...NODE_MASTERY].sort((a,b) => { if(a.score===null&&b.score===null)return 0; if(a.score===null)return 1; if(b.score===null)return -1; return b.score-a.score; });
  g.innerHTML = sorted.map(n => {
    let sc,scC,scT;
    if(n.score===null){sc='untrained';scC='none';scT='—';}
    else if(n.score>=0.85){sc='mastered';scC='high';scT=Math.round(n.score*100)+'%';}
    else if(n.score>=0.6){sc='learning';scC='mid';scT=Math.round(n.score*100)+'%';}
    else{sc='struggling';scC='low';scT=Math.round(n.score*100)+'%';}
    return `<div class="node-item" title="${n.category}"><div class="node-status ${sc}"></div><div class="node-name">${n.name}</div><div class="node-score ${scC}">${scT}</div></div>`;
  }).join('');
}

function renderTimeline() {
  const c = document.getElementById('timeline');
  c.innerHTML = LESSONS.map((l,i) => {
    const isLast = i===LESSONS.length-1;
    let dot = l.status==='complete'?'complete':l.status==='active'?'active':'pending';
    let scoreH = '';
    if(l.score!==null){const sc=l.score>=80?'good':l.score>=50?'ok':'pending-score';scoreH=`<div class="timeline-score ${sc}">${l.score}% exam score</div>`;}
    else if(l.status==='active'){scoreH=`<div class="timeline-score pending-score">exam pending</div>`;}
    return `<div class="timeline-item"><div class="timeline-line"><div class="timeline-dot ${dot}"></div>${!isLast?'<div class="timeline-connector"></div>':''}</div><div class="timeline-content"><div class="timeline-title">${l.title}</div><div class="timeline-meta">${l.prompts} prompts</div>${scoreH}</div></div>`;
  }).join('');
}

function renderHealthAlerts() {
  const c = document.getElementById('healthAlerts');
  if (!c || !HEALTH_ALERTS.length) { if(c) c.innerHTML='<div style="color:var(--text-dim);font-size:13px;padding:20px;">No health data yet. Run the health monitor after training.</div>'; return; }
  const filtered = HEALTH_ALERTS.filter(a => a.level !== 'INFO');
  const infoAlerts = HEALTH_ALERTS.filter(a => a.level === 'INFO');
  const all = [...filtered, ...infoAlerts];
  c.innerHTML = all.map(a => {
    return `<div class="health-alert-item alert-${a.level}"><div class="health-alert-title"><span class="alert-tag tag-${a.level}">${a.level}</span>${a.title}</div><div class="health-alert-detail">${a.detail}</div></div>`;
  }).join('');
}

renderChart(); renderNodes(); renderTimeline(); renderHealthAlerts();

// Fix error bar percentages
const errMax = {{ERR_MAX}};
document.querySelectorAll('.error-bar-fill').forEach(el => {
  const w = parseFloat(el.style.width);
  if(isNaN(w) || w === 0) el.style.width = '0%';
});
</script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(description='Generate BlueprintLLM training dashboard')
    parser.add_argument('--training-log', type=str, default=None, help='Path to training console output')
    parser.add_argument('--exam-dir', type=str, default='exams', help='Directory containing exam results')
    parser.add_argument('--lesson-dir', type=str, default='lessons', help='Directory containing lesson files')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory containing model configs')
    parser.add_argument('--output', type=str, default='dashboard/index.html', help='Output HTML file')
    parser.add_argument('--open', action='store_true', help='Open in browser after generating')
    args = parser.parse_args()

    plog.start_step("9.1", "Update dashboard")
    print("BlueprintLLM Dashboard Generator")
    print("=" * 40)

    # Auto-detect training log if not specified
    log_path = args.training_log
    if not log_path:
        log_path = detect_latest_training_log(args.model_dir, 'logs')
        if log_path:
            print(f"  Auto-detected training log: {log_path}")
        else:
            print("  No training log found (use --training-log to specify)")

    # Parse all data sources
    print("  Parsing training log...")
    training_data = parse_training_log(log_path)
    if training_data:
        print(f"    {len(training_data['steps'])} training steps, {len(training_data['evals'])} eval points")
        print(f"    Final loss: {training_data['final_loss']:.3f}, accuracy: {training_data['final_accuracy']*100:.1f}%")

    print("  Parsing exam results...")
    exams, node_scores = parse_exam_results(args.exam_dir)
    print(f"    {len(exams)} exams found, {len(node_scores)} nodes scored")

    print("  Parsing lessons...")
    lessons = parse_lessons(args.lesson_dir)
    print(f"    {len(lessons)} lessons found")

    # Build derived data
    node_mastery = build_node_mastery_list(node_scores)

    # Error breakdown from most recent exam
    error_breakdown = None
    if exams:
        latest = exams[-1]
        error_breakdown = latest.get('error_breakdown', latest.get('errors', {}))

    activity_log = build_activity_log(training_data, exams, lessons)

    # Load health report
    print("  Loading health report...")
    health_report = load_health_report()
    if health_report:
        overall = health_report.get('summary', {}).get('overall_health', '?')
        count = health_report.get('summary', {}).get('total_alerts', 0)
        print(f"    Health status: {overall}, {count} alerts")
    else:
        print("    No health report found (run 19_training_health_monitor.py first)")

    # Generate HTML
    print("  Generating dashboard...")
    html = generate_dashboard_html(training_data, node_mastery, lessons, exams, activity_log, error_breakdown, health_report)

    # Write output
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html)

    abs_path = os.path.abspath(args.output)
    print(f"\n  Dashboard saved to: {abs_path}")
    print(f"  Open in browser: file:///{abs_path.replace(os.sep, '/')}")

    if args.open:
        webbrowser.open(f'file:///{abs_path.replace(os.sep, "/")}')
        print("  Opened in browser!")

    print("\nDone!")
    plog.complete_step("9.1", "Update dashboard", abs_path)


if __name__ == '__main__':
    main()
