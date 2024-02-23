rule train_model:
    params:
        data_path = '/home/katya.govorkova/challenge_datasets/ligo_datasets/output'
    output:
        model = 'output/model.pth'
    shell:
        'python3 scripts/train.py {params.data_path} {output.model}'


rule evaluate_on_blackbox:
    input:
        model = rules.train_model.output.model
    params:
        data_path = '/home/katya.govorkova/challenge_datasets/ligo_datasets/output'
    output:
        submission = 'output/submission.npy'
    shell:
        'python3 scripts/evaluate.py {params.data_path} {input.model} {output.submission}'