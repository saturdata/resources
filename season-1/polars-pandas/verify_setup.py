#!/usr/bin/env python
"""Quick verification script to check notebook setup."""

from pathlib import Path
import sys


def verify_setup():
    """Verify all files exist and are ready."""
    print("🔍 Verifying Episode 1 setup...")
    print()

    # Check we're in the right directory
    cwd = Path.cwd()
    print(f"📂 Current directory: {cwd}")

    if not cwd.name == "polars-pandas":
        print("⚠️  Warning: You should be in the 'polars-pandas' directory")
        print(f"   Current: {cwd.name}")
        return False

    print("✅ Correct directory")
    print()

    # Check data files
    data_dir = cwd / "data"
    required_files = {
        "transactions.csv": 300_000_000,  # ~330MB
        "customers.csv": 6_000_000,  # ~6.6MB
        "products.csv": 400_000,  # ~444KB
    }

    print("📊 Checking data files...")
    all_good = True

    for filename, min_size in required_files.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"❌ Missing: {filename}")
            all_good = False
        else:
            size_mb = filepath.stat().st_size / 1024 / 1024
            if filepath.stat().st_size < min_size:
                print(f"⚠️  {filename}: {size_mb:.1f} MB (seems too small)")
            else:
                print(f"✅ {filename}: {size_mb:.1f} MB")

    if not all_good:
        print()
        print("💡 Generate missing data files:")
        print("   python data/generate_dataset.py")
        return False

    print()

    # Check notebook file
    notebook = cwd / "resource.py"
    if not notebook.exists():
        print("❌ Missing: resource.py")
        return False
    print(f"✅ Notebook: resource.py ({notebook.stat().st_size / 1024:.1f} KB)")

    print()
    print("🎉 All checks passed! Ready to run:")
    print("   uv run marimo run resource.py")
    return True


if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)
