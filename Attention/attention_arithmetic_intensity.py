"""
Simple arithmetic intensity calculator for standard attention.

Run:
    uv run python Attention/attention_arithmetic_intensity.py
"""

import argparse


def mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def calculate_attention_ai(N: int, d: int, bytes_per_value: int = 2) -> dict:
    qkv_bytes = N * d * bytes_per_value
    s_or_p_bytes = N * N * bytes_per_value
    o_bytes = N * d * bytes_per_value

    # Standard attention:
    # S = QK^T
    # P = softmax(S)
    # O = PV

    # For FP16/BF16 where bytes_per_value = 2:
    # total memory movement = 8N^2 + 8Nd bytes
    total_memory_bytes = 4 * bytes_per_value * N * N + 4 * bytes_per_value * N * d

    # total compute = 4N^2d + 3N^2 operations
    total_compute_ops = 4 * N * N * d + 3 * N * N

    arithmetic_intensity = total_compute_ops / total_memory_bytes

    return {
        "N": N,
        "d": d,
        "bytes_per_value": bytes_per_value,
        "Q_bytes": qkv_bytes,
        "K_bytes": qkv_bytes,
        "V_bytes": qkv_bytes,
        "S_bytes": s_or_p_bytes,
        "P_bytes": s_or_p_bytes,
        "O_bytes": o_bytes,
        "total_memory_bytes": total_memory_bytes,
        "total_compute_ops": total_compute_ops,
        "arithmetic_intensity": arithmetic_intensity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate arithmetic intensity for standard attention."
    )
    parser.add_argument("--N", type=int, default=4096, help="Sequence length")
    parser.add_argument("--d", type=int, default=128, help="Attention head dimension")
    parser.add_argument(
        "--bytes-per-value",
        type=int,
        default=2,
        help="FP16/BF16 = 2 bytes, FP32 = 4 bytes",
    )
    parser.add_argument(
        "--hardware-ops-per-byte",
        type=float,
        default=295.0,
        help="Reference GPU ops:byte ratio",
    )

    args = parser.parse_args()
    result = calculate_attention_ai(args.N, args.d, args.bytes_per_value)

    print()
    print("Standard Attention Arithmetic Intensity")
    print("=" * 52)

    print(f"N: {result['N']}")
    print(f"d: {result['d']}")
    print(f"bytes per value: {result['bytes_per_value']}")

    print()
    print("Matrix shapes")
    print("-" * 52)
    print(f"Q, K, V: {args.N} x {args.d}")
    print(f"S, P:    {args.N} x {args.N}")
    print(f"O:       {args.N} x {args.d}")

    print()
    print("Matrix memory")
    print("-" * 52)
    print(f"Q: {result['Q_bytes']:,} bytes = {mib(result['Q_bytes']):.2f} MiB")
    print(f"K: {result['K_bytes']:,} bytes = {mib(result['K_bytes']):.2f} MiB")
    print(f"V: {result['V_bytes']:,} bytes = {mib(result['V_bytes']):.2f} MiB")
    print(f"S: {result['S_bytes']:,} bytes = {mib(result['S_bytes']):.2f} MiB")
    print(f"P: {result['P_bytes']:,} bytes = {mib(result['P_bytes']):.2f} MiB")
    print(f"O: {result['O_bytes']:,} bytes = {mib(result['O_bytes']):.2f} MiB")

    print()
    print("Totals")
    print("-" * 52)
    print(
        f"Total memory movement: {result['total_memory_bytes']:,} bytes "
        f"= {mib(result['total_memory_bytes']):.2f} MiB"
    )
    print(f"Total compute: {result['total_compute_ops']:,} ops")
    print(f"Arithmetic intensity: {result['arithmetic_intensity']:.2f} ops/byte")

    print()
    print("Bottleneck verdict")
    print("-" * 52)
    if result["arithmetic_intensity"] < args.hardware_ops_per_byte:
        print(
            f"{result['arithmetic_intensity']:.2f} ops/byte "
            f"< {args.hardware_ops_per_byte:.2f} ops/byte"
        )
        print("Verdict: memory-bound")
    else:
        print(
            f"{result['arithmetic_intensity']:.2f} ops/byte "
            f">= {args.hardware_ops_per_byte:.2f} ops/byte"
        )
        print("Verdict: compute-bound")


if __name__ == "__main__":
    main()
