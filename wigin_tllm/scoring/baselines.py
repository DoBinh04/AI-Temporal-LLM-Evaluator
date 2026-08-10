"""Optional reference baselines for the SVD gate.

These are the published ChronoGPT reference models. They are *data*, not
policy: pass them to :class:`~wigin_tllm.scoring.svd_gate.SvdGate` when
you want submissions rejected for being lightly-modified copies of them,
or supply your own mapping (or none at all).

Each entry is pinned to a commit SHA so the baseline itself cannot drift.
"""

CHRONOGPT_BASELINES: dict[int, list[str]] = {
    2013: [
        "manelalab/chrono-gpt-v1-20131231@6f2e595689458b1809d5c6efb9a6095257347ca2",
        "manelalab/chrono-gpt-instruct-v1-20131231@f35f1596d860a797df1c592a5a70bf02a3a00884",
    ],
    2014: [
        "manelalab/chrono-gpt-v1-20141231@4fba07f4ef563b3addf2b05f385d0b347bf1cc0d",
        "manelalab/chrono-gpt-instruct-v1-20141231@e121db790ca77ebb082c025b2438717644ee1cfb",
    ],
    2015: [
        "manelalab/chrono-gpt-v1-20151231@aacd4c4e8020dd0ad686d36f18bcf34cd8003bc3",
        "manelalab/chrono-gpt-instruct-v1-20151231@5a7f3439fd5d782b3780c366160c43177e6f5eba",
    ],
    2016: [
        "manelalab/chrono-gpt-v1-20161231@20d93dc9b103644b212db413db4ab1207063d010",
        "manelalab/chrono-gpt-instruct-v1-20161231@5aec0aacc696f9526e12abe22a3fc96348dfca1d",
    ],
    2017: [
        "manelalab/chrono-gpt-v1-20171231@4cc4334a2c2d38ae35deb0bb7fcae642d3f73a10",
        "manelalab/chrono-gpt-instruct-v1-20171231@5f6b4ab1664bd5e658af44ad6b02183178b81b55",
    ],
    2018: [
        "manelalab/chrono-gpt-v1-20181231@17d7de7945199ff03be989ca84d00c0f59a975af",
        "manelalab/chrono-gpt-instruct-v1-20181231@331c03be137a1a80f1a371232d3d6a9636f6ad9a",
    ],
    2019: [
        "manelalab/chrono-gpt-v1-20191231@7e62517f31b11fad179c79ce79a465aa00c7ee4d",
        "manelalab/chrono-gpt-instruct-v1-20191231@4dfb7817915d07d0ed99815877186f827ec3b88e",
    ],
    2020: [
        "manelalab/chrono-gpt-v1-20201231@c0d2acbd2a378ac79d8a5ae79a9447d23145eb8a",
        "manelalab/chrono-gpt-instruct-v1-20201231@f8020c2c939645abbec9caf8a0cdd1d7806cb42a",
    ],
    2021: [
        "manelalab/chrono-gpt-v1-20211231@a070953708ee809e630e4d9652e9c753d7b6782e",
        "manelalab/chrono-gpt-instruct-v1-20211231@7f3c7d0dccea060d96dfb89391ef830655b8dbaf",
    ],
    2022: [
        "manelalab/chrono-gpt-v1-20221231@993711fdf078740fe1c837a3687528e2173443d2",
        "manelalab/chrono-gpt-instruct-v1-20221231@f1b8c4eb806a9fe7c26b7e5d30cf003304ed9281",
    ],
    2023: [
        "manelalab/chrono-gpt-v1-20231231@8ac22e54d37df8bb8037622680414118239fbe53",
        "manelalab/chrono-gpt-instruct-v1-20231231@2156f3ac9a36916773664266397682b951d43411",
    ],
    2024: [
        "manelalab/chrono-gpt-v1-20241231@1d9f1b8ff50bb45a6fe1402280e617af4c2d805c",
        "manelalab/chrono-gpt-instruct-v1-20241231@c162df20666475d125737e030943e18e10b3d91f",
    ],
}
