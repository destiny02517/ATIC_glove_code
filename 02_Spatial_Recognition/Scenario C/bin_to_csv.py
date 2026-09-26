#!python3

# *********************************************************
# Script to convert an InfiniiVision oscilloscope binary
# file to CSV format waveform files.
# *********************************************************

# =========================================================
# Import Modules
# =========================================================
import sys
import re
import string
import struct

# ---------------------------------------------------------
# Variables.
# ---------------------------------------------------------
# 定义输入文件名列表
input_files = [f"scope_{i}.bin" for i in range(33,53)]  # 生成 scope_0.bin 到 scope_8.bin 的文件名列表

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
def prtsv(string, msg):
    msg.write("%s\n" % string)
    print(string)


# =========================================================
# Function to read 8-bit digital data from the binary data
# file.
# =========================================================
def read_8bit_digital_data(bin_input, buffer_size, x_origin, x_increment, label, segment_index, input_file, msg):
    if segment_index == 0:
        csv_output_file = re.sub("\.bin", "_%s.csv" % label, input_file)
    else:
        csv_output_file = re.sub("\.bin", "_segment-%d_%s.csv" % (segment_index, label), input_file)
    csv = open(csv_output_file, "w")

    for i in range(buffer_size):
        (digital_data,) = struct.unpack('B', bin_input.read(1))
        hex_string = hex(digital_data)  # Returns "0xn" or "0xnn".
        if len(hex_string) == 4:  # "0xnn".
            un = hex_to_binary_dict[hex_string[2]]
            ln = hex_to_binary_dict[hex_string[3]]
        else:  # "0xn".
            un = "0000"
            ln = hex_to_binary_dict[hex_string[2]]
        csv.write("%s, %s, %s, %s, %s, %s, %s, %s, %s\n" % (
            x_origin + (i * x_increment), un[0], un[1], un[2], un[3], ln[0], ln[1], ln[2], ln[3], ))

    csv.close()
    prtsv(f"CSV waveform data saved to: {csv_output_file}", msg)


# =========================================================
# Function to read 32-bit float data from the binary data
# file and convert negative values to positive.
# =========================================================
def read_32bit_float_data(bin_input, buffer_size, bytes_per_point, x_origin, x_increment, label, segment_index, input_file, msg):
    if segment_index == 0:
        csv_output_file = re.sub("\.bin", "_%s.csv" % label, input_file)
    else:
        csv_output_file = re.sub("\.bin", "_segment-%d_%s.csv" % (segment_index, label), input_file)
    csv = open(csv_output_file, "w")

    for i in range(int(buffer_size / bytes_per_point)):
        (voltage,) = struct.unpack('f', bin_input.read(bytes_per_point))
        voltage = abs(voltage)  # 翻转负值为正值
        csv.write("%E, %f\n" % (x_origin + (i * x_increment), voltage))

    csv.close()
    prtsv(f"CSV waveform data saved to: {csv_output_file}", msg)


# =========================================================
# Function to print data from individual waveforms in binary
# data file.
# =========================================================
def read_waveform_data(bin_input, x_origin, x_increment, label, segment_index, input_file, msg):
    (waveform_data_header_size,) = struct.unpack('i', bin_input.read(4))
    (buffer_type,) = struct.unpack('h', bin_input.read(2))
    (bytes_per_point,) = struct.unpack('h', bin_input.read(2))
    (buffer_size,) = struct.unpack('i', bin_input.read(4))

    if buffer_type == 1:  # Normal 32-bit float data.
        read_32bit_float_data(bin_input, buffer_size, bytes_per_point, x_origin, x_increment, label, segment_index, input_file, msg)
    elif buffer_type == 2:  # Maximum float data.
        label = label + "_PkMax"
        read_32bit_float_data(bin_input, buffer_size, bytes_per_point, x_origin, x_increment, label, segment_index, input_file, msg)
    elif buffer_type == 3:  # Minimum float data.
        label = label + "_PkMin"
        read_32bit_float_data(bin_input, buffer_size, bytes_per_point, x_origin, x_increment, label, segment_index, input_file, msg)
    elif buffer_type == 6:  # Digital unsigned 8-bit char data.
        read_8bit_digital_data(bin_input, buffer_size, x_origin, x_increment, label, segment_index, input_file, msg)
    else:
        buffer_bytes = bin_input.read(buffer_size)


# =========================================================
# Function to print data from individual waveforms in binary
# data file.
# =========================================================
def read_waveform(bin_input, input_file, msg):
    (waveform_header_size,) = struct.unpack('i', bin_input.read(4))
    (waveform_type,) = struct.unpack('i', bin_input.read(4))
    (waveform_buffers,) = struct.unpack('i', bin_input.read(4))
    (points,) = struct.unpack('i', bin_input.read(4))
    (count,) = struct.unpack('i', bin_input.read(4))
    (x_display_range,) = struct.unpack('f', bin_input.read(4))
    (x_display_origin,) = struct.unpack('d', bin_input.read(8))
    (x_increment,) = struct.unpack('d', bin_input.read(8))
    (x_origin,) = struct.unpack('d', bin_input.read(8))
    (x_units,) = struct.unpack('i', bin_input.read(4))
    (y_units,) = struct.unpack('i', bin_input.read(4))
    (date,) = struct.unpack('16s', bin_input.read(16))
    (time,) = struct.unpack('16s', bin_input.read(16))
    (frame,) = struct.unpack('24s', bin_input.read(24))
    (waveform_label,) = struct.unpack('16s', bin_input.read(16))
    label = waveform_label.decode("utf-8").rstrip(chr(0))
    (time_tags,) = struct.unpack('d', bin_input.read(8))
    (segment_index,) = struct.unpack('I', bin_input.read(4))

    for i in range(waveform_buffers):
        read_waveform_data(bin_input, x_origin, x_increment, label, segment_index, input_file, msg)


# =========================================================
# Main Program
# =========================================================
for input_file in input_files:
    # Open message output file.
    msg_output_file = re.sub("\.bin", "_info.txt", input_file)
    msg = open(msg_output_file, "w")

    # Open binary file.
    try:
        bin_input = open(input_file, "rb")
    except FileNotFoundError:
        prtsv(f"File not found: {input_file}. Skipping...", msg)
        continue

    prtsv(f"Processing file: {input_file}", msg)

    (cookie,) = struct.unpack('2s', bin_input.read(2))
    (file_version,) = struct.unpack('2s', bin_input.read(2))
    (file_size,) = struct.unpack('i', bin_input.read(4))
    (waveforms,) = struct.unpack('i', bin_input.read(4))

    for i in range(waveforms):
        read_waveform(bin_input, input_file, msg)

    # Close binary file.
    bin_input.close()

    # Close message output file.
    msg.close()

# Exit program.
sys.exit()