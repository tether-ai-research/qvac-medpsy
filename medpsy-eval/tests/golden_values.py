"""Pinned end-to-end golden values for every suite case.

Do not regenerate these to make a failing test pass: a mismatch means behavior changed.
"""

EXPECTED: dict[str, dict] = {
    'closed_ended_v1/AfriMedQA': {
        'name': 'AfriMedQA',
        'dataset_samples': 2,
        'epochs': 3,
        'reducer': [
            'mean',
        ],
        'solvers': [
            'generate',
        ],
        'scorers': [
            'judge_extraction',
            'generation_stats',
        ],
        'remote': [
            '["load_dataset", "afrimedqa/afrimedqa_v2", [], {"revision": "3b4382fa0bb51bfc026f5813021ab0ec7be9de8f", "split": "train"}]',
        ],
        'task_metadata': '1c621ff8873f37340544f3f5b333e29a00e1ed3c23445db7e1c7a1cb4e912adf',
        'inputs': '1c0eb655385943c65383409fa7f0b6157e1c108798a5c0ed56a7d1ba91eadbf3',
        'model_calls': '4b6a4384ddae8056cbe9ebfe9c78e51aab9d4f1f79ba4324580f31ebdce1029a',
        'scores': '1185d3ff7f8f255abe72fb0895c0dcdfde7e09e347b6dcb71bd15006337c3d44',
        'metrics': '29483d14dceeb5a0b399eefb4fbf081c744f766816554252e2d1e62d368ee7e0',
    },
    'closed_ended_v1/MedQA_US': {
        'name': 'MedQA_US',
        'dataset_samples': 2,
        'epochs': 3,
        'reducer': [
            'mean',
        ],
        'solvers': [
            'generate',
        ],
        'scorers': [
            'judge_extraction',
            'generation_stats',
        ],
        'remote': [
            '["load_dataset", "xuxuxuxuxu/MedQA_US_test", [], {"revision": "f2b84f2b44f6fccd6872e8bdab99708290826456", "split": "train"}]',
        ],
        'task_metadata': '1c621ff8873f37340544f3f5b333e29a00e1ed3c23445db7e1c7a1cb4e912adf',
        'inputs': 'e7a9c4e032621bee1d37e59a9c0ce1b5a1dd2f6fd4fe6c515d0e0b5a19b845f3',
        'model_calls': '98e77ddd8d9455b69a4ec1692d99028db8fd586672010b0c3082600bcbb459ee',
        'scores': '0302e4887cc430c7515a6783ec1a2fea8e6ef3f1399e47fee5c05f25ebc91f94',
        'metrics': '2e2ae2db7a4f15518073c35b980426cd22536f763fd42b33fbf9f4b941172160',
    },
    'closed_ended_v1/PubMedQA': {
        'name': 'PubMedQA',
        'dataset_samples': 3,
        'epochs': 3,
        'reducer': [
            'mean',
        ],
        'solvers': [
            'generate',
        ],
        'scorers': [
            'judge_extraction',
            'generation_stats',
        ],
        'remote': [
            '["hf_hub_download", "bigbio/pubmed_qa", "pqal.zip", {"repo_type": "dataset", "revision": "15578e80ef02c2c9636d33fdd7eb6b05b987ba60"}]',
        ],
        'task_metadata': '1c621ff8873f37340544f3f5b333e29a00e1ed3c23445db7e1c7a1cb4e912adf',
        'inputs': 'e8a2ae9a933500ffb87f4f75186449ae16a4c45af11c3189647c513bfd607ff2',
        'model_calls': '2d461c5dd78f1532762cf028b5b7b34c74f330e74a795814f95798a553072414',
        'scores': 'eb17ad867e563e86769cb8037ee90b22d42550e0438d5db944863d01572ce792',
        'metrics': '2b09937b1505e5fd2f683a506d0b60d70abb91899f22c678ad27840c25efdc63',
    },
    'closed_ended_v1/mmlu': {
        'name': 'mmlu',
        'dataset_samples': 12,
        'epochs': 3,
        'reducer': [
            'mean',
        ],
        'solvers': [
            'generate',
        ],
        'scorers': [
            'judge_extraction',
            'generation_stats',
        ],
        'remote': [
            '["load_dataset", "cais/mmlu", ["anatomy"], {"revision": "c30699e8356da336a370243923dbaf21066bb9fe", "split": "test"}]',
            '["load_dataset", "cais/mmlu", ["clinical_knowledge"], {"revision": "c30699e8356da336a370243923dbaf21066bb9fe", "split": "test"}]',
            '["load_dataset", "cais/mmlu", ["college_biology"], {"revision": "c30699e8356da336a370243923dbaf21066bb9fe", "split": "test"}]',
            '["load_dataset", "cais/mmlu", ["college_medicine"], {"revision": "c30699e8356da336a370243923dbaf21066bb9fe", "split": "test"}]',
            '["load_dataset", "cais/mmlu", ["medical_genetics"], {"revision": "c30699e8356da336a370243923dbaf21066bb9fe", "split": "test"}]',
            '["load_dataset", "cais/mmlu", ["professional_medicine"], {"revision": "c30699e8356da336a370243923dbaf21066bb9fe", "split": "test"}]',
        ],
        'task_metadata': '1c621ff8873f37340544f3f5b333e29a00e1ed3c23445db7e1c7a1cb4e912adf',
        'inputs': '3b80089404d0fc31a743a19c4639e087a86f0eb71d0c0b81af61b819f804119f',
        'model_calls': '9e56a0098b4edb60fb305a4251e461ea89c5e2d0cb9816e826443e710c1fb315',
        'scores': 'd0986a09be7dac35eac5a8bb40ced0385ec5455009cc9664223a571618f1a548',
        'metrics': '8e58bd72d6b1004d8a0df33199b84f493cb5d4af64524c153c1f1fefd951a859',
    },
    'closed_ended_v1/medmcqa': {
        'name': 'medmcqa',
        'dataset_samples': 2,
        'epochs': 3,
        'reducer': [
            'mean',
        ],
        'solvers': [
            'generate',
        ],
        'scorers': [
            'judge_extraction',
            'generation_stats',
        ],
        'remote': [
            '["load_dataset", "openlifescienceai/medmcqa", [], {"revision": "91c6572c454088bf71b679ad90aa8dffcd0d5868", "split": "validation"}]',
        ],
        'task_metadata': '1c621ff8873f37340544f3f5b333e29a00e1ed3c23445db7e1c7a1cb4e912adf',
        'inputs': '368bba93e6c53f4df970c0e1da8fcaea53b75eed39cbc413398c091e0220d690',
        'model_calls': '19fa86cf0f249ec067916bd4d9d1c72b31c42154abf894adcbeedd80d4018829',
        'scores': '5bcc95c213019147b2de111472fd6392f72a0fc1460339b7fb557676632853e1',
        'metrics': 'd06ad779f429ef9710068afb085eeb656fed4fd7a9f6e08f8049967d824eb4df',
    },
    'closed_ended_v1/medxpertqa': {
        'name': 'medxpertqa',
        'dataset_samples': 2,
        'epochs': 3,
        'reducer': [
            'mean',
        ],
        'solvers': [
            'generate',
        ],
        'scorers': [
            'judge_extraction',
            'generation_stats',
        ],
        'remote': [
            '["load_dataset", "TsinghuaC3I/MedXpertQA", ["Text"], {"revision": "7e7c465a68eb2b866926bfa59c8c9d17a8daba65", "split": "test"}]',
        ],
        'task_metadata': '1c621ff8873f37340544f3f5b333e29a00e1ed3c23445db7e1c7a1cb4e912adf',
        'inputs': '2d5d764c617938ee327cdd7a6d001ae66e826432ac0d9f595d7dbf9c7331652d',
        'model_calls': 'c64d8be2853f7de04e2e9762087393034c683ddc659a70bcea91f905ccb9f0f9',
        'scores': '421d303e0e1b79cd806a3a3adec62ad664ee2aa41b50b4235934753b0bfb4fc4',
        'metrics': '404ef9caf1ef8bd680cd6cb08ce8cbac403373ac8cddd28cf2c5902c746b85fd',
    },
    'closed_ended_v1/mmlu_pro_health': {
        'name': 'mmlu_pro_health',
        'dataset_samples': 2,
        'epochs': 3,
        'reducer': [
            'mean',
        ],
        'solvers': [
            'generate',
        ],
        'scorers': [
            'judge_extraction',
            'generation_stats',
        ],
        'remote': [
            '["load_dataset", "TIGER-Lab/MMLU-Pro", [], {"revision": "7db2dc66c66870201e1cf041a3a9297a7674c50a", "split": "test"}]',
        ],
        'task_metadata': '1c621ff8873f37340544f3f5b333e29a00e1ed3c23445db7e1c7a1cb4e912adf',
        'inputs': 'ecdcc5b127d35abc881bb5261d8238a87c335ca49176894472b0684fdc25a32b',
        'model_calls': '8fb2ad0792fcbdd50304194e97241fdec8e42a3b9f40c3bc13884a23a8e2abf5',
        'scores': '24a79557d2b20bcc0bb9f16ffa96fa69fa5bb76f493ea308dcd3b0fc9678f283',
        'metrics': '8f566aa2392fbdbe1efef14b9f7e8da4b68bbaf265b52d705157f8ff9f0c5fd3',
    },
    'healthbench_v1': {
        'name': 'healthbench',
        'dataset_samples': 3,
        'epochs': 1,
        'reducer': [
            'mean',
        ],
        'solvers': [
            'generate',
        ],
        'scorers': [
            'healthbench_rubric',
            'generation_stats',
        ],
        'remote': [
            '["load_dataset", "huihuixu/healthbench", [], {"data_files": {"test": "hard_2025-05-08-21-00-10.jsonl"}, "revision": "e35c21171476f5e25803f936c6d48eb3df97b65e", "split": "test"}]',
            '["load_dataset", "huihuixu/healthbench", [], {"data_files": {"test": "consensus_2025-05-09-20-00-46.jsonl"}, "revision": "e35c21171476f5e25803f936c6d48eb3df97b65e", "split": "test"}]',
            '["load_dataset", "huihuixu/healthbench", [], {"data_files": {"test": "2025-05-07-06-14-12_oss_eval.jsonl"}, "revision": "e35c21171476f5e25803f936c6d48eb3df97b65e", "split": "test"}]',
        ],
        'task_metadata': '47e294a4da00e0d94ed696b9801b8267ee4bab48876913130c446b98cc83e2ca',
        'inputs': '19e7ce2167ae8e5827f8eb5833b8eae5c282632b47c5f61d3290b0368e232020',
        'model_calls': '9e6a81f5df4cec75414e535dee3e49b76bf4f78ed560aa1d8b56cdfb3be18cb6',
        'scores': '6f8b55c69af0ed52cf06a1ea1541321c81dca89806a205bc0b6f6699613a49b6',
        'metrics': '5cc51047849259483ec89a3c1ef1c0e9351b0eb7857549d0908e0737b4f0b3dd',
    },
}
