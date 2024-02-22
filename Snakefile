rule train_model:
    input:
    output:
        model = directory('output/saved_model')
    shell:
        'python3 scripts/train.py {output.model}'

rule evaluate_on_blackbox:
    input:
        rules.train_model.output.model
    output:
    shell:
        'python3 scripts/evaluate.py'