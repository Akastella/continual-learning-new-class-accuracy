"""Real prefix/resource witness using official TRAIN images only, no external data."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_r13_early_retention as r


def main():
    start = time.monotonic()
    out = ROOT / 'audit/r13_early_retention/trainonly_resource'
    out.mkdir(parents=True, exist_ok=False)
    environment = r.initialize_environment()
    rows = []
    for run in r.runs()[:2]:
        cfg = r.read(ROOT/run['config_path'])
        x, y, proxy, provenance = r.training.load_train_only(ROOT/'data', cfg, 4)
        initial = r.torch.load(ROOT/run['output']/'pre_task_004.pt', map_location='cpu', weights_only=True)
        model = r.engine.local_model(cfg, cfg['init_seed'], 'cuda', initial)
        matched = []

        def callback(step, net, path):
            if step in r.OVERLAP:
                old = r.state(ROOT/r.OLD/run['id']/f'checkpoints/step_{step:06d}.pt', 'probe')
                assert all(r.torch.equal(v.cpu(), old[k]) for k, v in net.state_dict().items())
                matched.append(step)
            else:
                r.inference(net, proxy[:1000])

        summary = r.training.train_trajectory(model, x, y, cfg, 4, out/run['id'],
            list(r.core.DENSE), callback=callback, stop_step=16)
        assert matched == r.OVERLAP
        model.load_state_dict(r.state(ROOT/run['output']/'post_task_004.pt', 'actual'))
        r.inference(model, proxy[:1000])
        for _ in range(7):
            r.inference(model, proxy)
        rows.append(dict(run_id=run['id'], optimizer_updates=summary['optimizer_steps_executed'],
            matched_steps=matched, train_only_proxy=True, validation_proxy_images=12000,
            test_proxy_images=14000, training_summary_seconds=summary['seconds'],
            order_sha256=summary['batch_order_sha256']))
        del model, x, y, proxy
    elapsed = time.monotonic()-start
    projection = elapsed*10*2
    result = dict(status='passed_train_only_prefix_and_resource_witness', environment=environment,
        decoded_data_splits=['official_CIFAR10_train'], external_model_access=False,
        rows=rows, elapsed_seconds=elapsed, full_matrix_projection_with_2x_margin_seconds=projection,
        within_3300_seconds=projection < 3300,
        source_sha256={p:r.sha(ROOT/p) for p in ['tools/measure_r13_trainonly.py',
            'tools/run_r13_early_retention.py', 'experiments/r13_early_retention/core.py',
            'experiments/r9_retention_trajectory/core.py']},
        scope='Cost proxy, exact six-prefix CUDA identity for two fixed histories; not scientific external evaluation or full-pipeline runtime guarantee')
    r.save(out/'RESOURCE.json', result)
    print(result)
    if not result['within_3300_seconds']:
        raise RuntimeError('Resource gate failed')


if __name__ == '__main__':
    main()
