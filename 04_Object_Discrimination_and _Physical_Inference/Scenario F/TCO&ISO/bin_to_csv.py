#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# *********************************************************
# Script to convert an InfiniiVision oscilloscope binary
# file to CSV format waveform files.
# *********************************************************

# =========================================================
# Import Modules
# =========================================================
import sys
import re
import struct
from pathlib import Path

# ---------------------------------------------------------
# Folders (relative to current working directory)
# ---------------------------------------------------------
INPUT_DIR = Path.cwd() / "raw data"
OUTPUT_DIR = Path.cwd() / "Data proceed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Define input files: read all .bin files in "raw data"
# ---------------------------------------------------------
input_files = sorted(INPUT_DIR.glob("*.bin"))
# 如果你只想处理 scope_*.bin，可以用下面这行替换上一行：
# input_files = sorted(INPUT_DIR.glob("scope_*.bin"))

# ---------------------------------------------------------
# Dictionaries (kept as-is)
# ---------------------------------------------------------
waveform_type_dict = {
    0: "Unknown",
    1: "Normal",
    2: "Peak Detect",
    3: "Average",
    4: "Horizontal Histogram",
    5: "Vertical Histogram",
    6: "Logic",
}
buffer_type_dict = {
    0: "Unknown data",
    1: "Normal 32-bit float data",
    2: "Maximum float data",
    3: "Minimum float data",
    4: "Time float data",
    5: "Counts 32-bit float data",
    6: "Digital unsigned 8-bit character data",
}
units_dict = {
    0: "Unknown",
    1: "Volts",
    2: "Seconds",
    3: "Constant",
    4: "Amps",
    5: "dB",
    6: "Hz",
}
hex_to_binary_dict = {
    "0": "0000",
    "1": "0001",
    "2": "0010",
    "3": "0011",
    "4": "0100",
    "5": "0101",
    "6": "0110",
    "7": "0111",
    "8": "1000",
    "9": "1001",
    "a": "1010",
    "b": "1011",
    "c": "1100",
    "d": "1101",
    "e": "1110",
    "f": "1111",
}

# =========================================================
# Function to print and save output information.
# =========================================================
def prtsv(text, msg):
    msg.write(f"{text}\n")
    print(text)

def safe_label(label: str) -> str:
    # 避免 label 里带空格/斜杠等导致文件名异常
    label = label.strip()
    label = re.sub(r"[^\w\-.]+", "_", label)  # 非字母数字下划线点横线 -> _
    return label if label else "CH"

# =========================================================
# Read 8-bit digital data and save to CSV in OUTPUT_DIR
# =========================================================
def read_8bit_digital_data(
    bin_input, buffer_size, x_origin, x_increment,
    label, segment_index, input_path: Path, msg
):
    label = safe_label(label)
    stem = input_path.stem

    if segment_index == 0:
        out_name = f"{stem}_{label}.csv"
    else:
        out_name = f"{stem}_segment-{segment_index}_{label}.csv"

    csv_output_path = OUTPUT_DIR / out_name
    with open(csv_output_path, "w", encoding="utf-8") as csv:
        for i in range(buffer_size):
            (digital_data,) = struct.unpack("B", bin_input.read(1))
            hex_string = hex(digital_data)  # "0xn" or "0xnn"
            if len(hex_string) == 4:  # "0xnn"
                un = hex_to_binary_dict[hex_string[2]]
                ln = hex_to_binary_dict[hex_string[3]]
            else:  # "0xn"
                un = "0000"
                ln = hex_to_binary_dict[hex_string[2]]

            csv.write(
                "%s, %s, %s, %s, %s, %s, %s, %s, %s\n"
                % (
                    x_origin + (i * x_increment),
                    un[0], un[1], un[2], un[3],
                    ln[0], ln[1], ln[2], ln[3],
                )
            )

    prtsv(f"CSV waveform data saved to: {csv_output_path}", msg)

# =========================================================
# Read 32-bit float data and save to CSV in OUTPUT_DIR
# =========================================================
def read_32bit_float_data(
    bin_input, buffer_size, bytes_per_point, x_origin, x_increment,
    label, segment_index, input_path: Path, msg
):
    label = safe_label(label)
    stem = input_path.stem

    if segment_index == 0:
        out_name = f"{stem}_{label}.csv"
    else:
        out_name = f"{stem}_segment-{segment_index}_{label}.csv"

    csv_output_path = OUTPUT_DIR / out_name
    with open(csv_output_path, "w", encoding="utf-8") as csv:
        npts = int(buffer_size / bytes_per_point)
        for i in range(npts):
            (voltage,) = struct.unpack("f", bin_input.read(bytes_per_point))
            voltage = abs(voltage)  # 翻转负值为正值
            csv.write("%E, %f\n" % (x_origin + (i * x_increment), voltage))

    prtsv(f"CSV waveform data saved to: {csv_output_path}", msg)

# =========================================================
# Read waveform buffer data
# =========================================================
def read_waveform_data(bin_input, x_origin, x_increment, label, segment_index, input_path: Path, msg):
    (waveform_data_header_size,) = struct.unpack("i", bin_input.read(4))
    (buffer_type,) = struct.unpack("h", bin_input.read(2))
    (bytes_per_point,) = struct.unpack("h", bin_input.read(2))
    (buffer_size,) = struct.unpack("i", bin_input.read(4))

    if buffer_type == 1:  # Normal 32-bit float data.
        read_32bit_float_data(bin_input, buffer_size, bytes_per_point, x_origin, x_increment, label, segment_index, input_path, msg)
    elif buffer_type == 2:  # Maximum float data.
        read_32bit_float_data(bin_input, buffer_size, bytes_per_point, x_origin, x_increment, label + "_PkMax", segment_index, input_path, msg)
    elif buffer_type == 3:  # Minimum float data.
        read_32bit_float_data(bin_input, buffer_size, bytes_per_point, x_origin, x_increment, label + "_PkMin", segment_index, input_path, msg)
    elif buffer_type == 6:  # Digital unsigned 8-bit char data.
        read_8bit_digital_data(bin_input, buffer_size, x_origin, x_increment, label, segment_index, input_path, msg)
    else:
        # Skip unknown buffer types
        bin_input.read(buffer_size)

# =========================================================
# Read waveform header + all buffers
# =========================================================
def read_waveform(bin_input, input_path: Path, msg):
    (waveform_header_size,) = struct.unpack("i", bin_input.read(4))
    (waveform_type,) = struct.unpack("i", bin_input.read(4))
    (waveform_buffers,) = struct.unpack("i", bin_input.read(4))
    (points,) = struct.unpack("i", bin_input.read(4))
    (count,) = struct.unpack("i", bin_input.read(4))
    (x_display_range,) = struct.unpack("f", bin_input.read(4))
    (x_display_origin,) = struct.unpack("d", bin_input.read(8))
    (x_increment,) = struct.unpack("d", bin_input.read(8))
    (x_origin,) = struct.unpack("d", bin_input.read(8))
    (x_units,) = struct.unpack("i", bin_input.read(4))
    (y_units,) = struct.unpack("i", bin_input.read(4))
    (date,) = struct.unpack("16s", bin_input.read(16))
    (time,) = struct.unpack("16s", bin_input.read(16))
    (frame,) = struct.unpack("24s", bin_input.read(24))
    (waveform_label,) = struct.unpack("16s", bin_input.read(16))
    label = waveform_label.decode("utf-8", errors="ignore").rstrip(chr(0))
    (time_tags,) = struct.unpack("d", bin_input.read(8))
    (segment_index,) = struct.unpack("I", bin_input.read(4))

    for _ in range(waveform_buffers):
        read_waveform_data(bin_input, x_origin, x_increment, label, segment_index, input_path, msg)

# =========================================================
# Main Program
# =========================================================
if not INPUT_DIR.exists():
    print(f"[ERROR] Input folder not found: {INPUT_DIR}")
    sys.exit(1)

if len(input_files) == 0:
    print(f"[WARN] No .bin files found in: {INPUT_DIR}")
    sys.exit(0)

for input_path in input_files:
    # message output file saved into OUTPUT_DIR
    msg_output_path = OUTPUT_DIR / f"{input_path.stem}_info.txt"
    with open(msg_output_path, "w", encoding="utf-8") as msg:
        # open binary
        try:
            bin_input = open(input_path, "rb")
        except FileNotFoundError:
            prtsv(f"File not found: {input_path}. Skipping...", msg)
            continue

        prtsv(f"Processing file: {input_path}", msg)

        (cookie,) = struct.unpack("2s", bin_input.read(2))
        (file_version,) = struct.unpack("2s", bin_input.read(2))
        (file_size,) = struct.unpack("i", bin_input.read(4))
        (waveforms,) = struct.unpack("i", bin_input.read(4))

        for _ in range(waveforms):
            read_waveform(bin_input, input_path, msg)

        bin_input.close()

print(f"Done. Outputs saved in: {OUTPUT_DIR}")
sys.exit(0)
