"""The example study: the experiment over two scales and two seeds."""

from tslab.lab.study import Grid, setting_axis, Study

from {{package_name}}.experiment import ExperimentConfig

study = Study(
    name='example',
    grids={
        'main': Grid(
            short_name='main',
            command=('uv', 'run', 'python', '-m', '{{package_name}}.experiment'),
            config=ExperimentConfig,
            axes={
                'scale': setting_axis(path='scale', values=(1.0, 2.0)),
                'seed': setting_axis(path='seed', values=(0, 1)),
            },
            run_path='{grid}/{scale}/seed_{seed}',
        )
    },
)
