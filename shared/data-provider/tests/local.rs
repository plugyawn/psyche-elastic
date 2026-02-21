use std::path::PathBuf;

use pretty_assertions::assert_eq;
use psyche_core::{BatchId, Shuffle, TokenSize};
use psyche_data_provider::{
    LengthKnownDataProvider, LocalDataProvider, LocalDataSplit, TokenizedDataProvider,
};
use tokenizers::Tokenizer;
use tokio::fs::read_to_string;

fn test_path(path: &[&str]) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "tests"]
        .iter()
        .chain(path)
        .collect()
}

const SEED: [u8; 32] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32,
];

#[tokio::test]
async fn loads_dolma_subset() {
    let data_dir = test_path(&["resources", "dolma", "data"]);
    let mut data_loader = LocalDataProvider::new_from_directory(
        data_dir,
        TokenSize::TwoBytes,
        2048,
        Shuffle::Seeded(SEED),
    )
    .unwrap();
    let samples = data_loader
        .get_samples(BatchId((0, 1).into()))
        .await
        .unwrap();

    let tokenizer = Tokenizer::from_file(test_path(&["resources", "llama2_tokenizer.json"]))
        .expect("tokenizer json exists");
    for (i, sample) in samples.into_iter().enumerate() {
        let decoded_path = test_path(&["resources", "dolma", "decoded", &format!("{i}.txt")]);

        let expected = read_to_string(&decoded_path)
            .await
            .unwrap_or_else(|_| panic!("no decoded file at {decoded_path:?}"));

        let decoded = tokenizer
            .decode(
                &sample
                    .input_ids
                    .into_iter()
                    .map(|x| x as u32)
                    .collect::<Vec<_>>(),
                true,
            )
            .unwrap();

        assert_eq!(
            decoded, expected,
            "sample {i} (left) doesn't match decoded reference (right) from file {decoded_path:?}"
        );
    }
}

#[tokio::test]
async fn loads_fineweb_subset() {
    let data_dir = test_path(&["resources", "fineweb", "data"]);
    let mut data_loader = LocalDataProvider::new_from_directory(
        data_dir,
        TokenSize::TwoBytes,
        2048,
        Shuffle::Seeded(SEED),
    )
    .unwrap();
    let samples = data_loader
        .get_samples(BatchId((0, 1).into()))
        .await
        .unwrap();

    let tokenizer = Tokenizer::from_file(test_path(&["resources", "llama2_tokenizer.json"]))
        .expect("tokenizer json exists");
    for (i, sample) in samples.into_iter().enumerate() {
        let decoded_path = test_path(&["resources", "fineweb", "decoded", &format!("{i}.txt")]);

        let expected = read_to_string(&decoded_path)
            .await
            .unwrap_or_else(|_| panic!("no decoded file at {decoded_path:?}"));

        let decoded = tokenizer
            .decode(
                &sample
                    .input_ids
                    .into_iter()
                    .map(|x| x as u32)
                    .collect::<Vec<_>>(),
                true,
            )
            .unwrap();

        assert_eq!(
            decoded, expected,
            "sample {i} (left) doesn't match decoded reference (right) from file {decoded_path:?}"
        );
    }
}

#[tokio::test]
async fn local_split_train_and_validation_files() {
    let data_dir = tempfile::tempdir().expect("tempdir");
    let source = test_path(&["resources", "fineweb", "data", "00000_00000_shuffled.ds"]);

    std::fs::copy(&source, data_dir.path().join("fineweb_train_000001.ds")).expect("copy train");
    std::fs::copy(&source, data_dir.path().join("fineweb_val_000000.ds")).expect("copy val");

    let train = LocalDataProvider::new_from_directory_with_split(
        data_dir.path(),
        TokenSize::TwoBytes,
        2048,
        Shuffle::DontShuffle,
        LocalDataSplit::Train,
    )
    .expect("train split");
    let val = LocalDataProvider::new_from_directory_with_split(
        data_dir.path(),
        TokenSize::TwoBytes,
        2048,
        Shuffle::DontShuffle,
        LocalDataSplit::Validation,
    )
    .expect("validation split");
    let all = LocalDataProvider::new_from_directory_with_split(
        data_dir.path(),
        TokenSize::TwoBytes,
        2048,
        Shuffle::DontShuffle,
        LocalDataSplit::All,
    )
    .expect("all split");

    assert!(train.num_sequences() > 0);
    assert!(val.num_sequences() > 0);
    assert_eq!(
        all.num_sequences(),
        train.num_sequences() + val.num_sequences()
    );

    let mut train = train;
    let mut val = val;
    let train_sample = train.get_samples(BatchId((0, 0).into())).await.unwrap();
    let val_sample = val.get_samples(BatchId((0, 0).into())).await.unwrap();
    assert_eq!(train_sample[0].input_ids.len(), 2049);
    assert_eq!(val_sample[0].input_ids.len(), 2049);
}
