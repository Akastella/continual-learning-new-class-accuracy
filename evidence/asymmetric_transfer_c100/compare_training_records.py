"""Compare the retained final-task training records with their original pairs."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def main():
    execution = read('execution.json')
    assert execution['status'] == 'complete'
    for key, name in [('plan_sha256', 'recorded_design.md'),
                      ('run_sha256', 'executed_training.py')]:
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == execution[key]
    inputs = {r['seed']: r for r in read('inputs.json')['seeds']}
    trained = read('training_complete.json')['records']
    fields = ('epoch', 'lr', 'next_lr', 'optimizer_steps', 'current_presentations',
              'replay_presentations', 'current_order_sha256', 'memory_indices_sha256',
              'replay_order_sha256', 'augmentation_keys_sha256')
    for current in trained:
        original = inputs[current['seed']]
        assert current['start_state_sha256'] == original['pre_state_sha256']
        assert current['start_state_sha256'] == original['actual']['start_state_sha256']
        assert current['start_state_sha256'] == original['common']['start_state_sha256']
        assert current['epochs'] == len(current['epoch_records']) == 70
        assert current['optimizer_steps'] == 5530
        assert current['current_presentations'] == current['replay_presentations'] == 350000
        for a, b in zip(current['epoch_records'], original['actual']['epoch_records']):
            assert all(a[k] == b[k] for k in fields)
    assert len(trained) == 5
    print(json.dumps({'status': 'passed', 'paired_seeds': 5, 'paired_epochs': 350,
                      'recorded_design_matches_execution': True,
                      'executed_source_matches_execution': True,
                      'large_model_states_checked_by_this_command': False}))

if __name__ == '__main__':
    main()
