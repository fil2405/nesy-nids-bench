from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw/nsl-kdd")
FILES = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"}


COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CLASS_TO_LABELS = {
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm", "processtable", "mailbomb"],
    "Probe": ["nmap", "portsweep", "ipsweep", "satan", "mscan", "saint"],
    "R2L": ["spy", "phf", "multihop", "ftp_write", "imap", "warezmaster", "warezclient", "guess_passwd", "sendmail", "named", "snmpget", "snmpguess", "xlock", "xsnoop", "httptunnel"],
    "U2R": ["perl", "buffer_overflow", "rootkit", "loadmodule", "xterm", "ps", "sqlattack"],
    "Normal": ["normal"]
} 

LABEL_TO_CLASS = {
    label: cls
    for cls, labels in CLASS_TO_LABELS.items()
    for label in labels
}

def load_nslkdd(split="train"):
    """Load NSL-KDD..."""

    if split not in FILES:
        raise ValueError(f"split must be one of {list(FILES)}, got {split!r}")
    
    df = pd.read_csv(DATA_DIR / FILES[split], names=COLUMNS)
    df["class"] = df["label"].map(LABEL_TO_CLASS)
    unknown = sorted(df.loc[df["class"].isna(), "label"].unique())
    
    if unknown:
        raise ValueError(f"Labels missing from CLASS_TO_LABELS: {unknown}")
    return df