import math
import subprocess
import os

# Dataset paths

# Example: this is stage 1 where and we cut at 2.6T tokens 
# Other stages mixing files and mixing coefficients can be found at:
# https://github.com/huggingface/smollm/tree/28490d016eabc9f81ab8b6f71c240eab1ebcdd27/text/pretraining
#
# TODO: we should consider adding an offset based on the alredy processed tokens

datasets = [
    "s3://smollm3/datasets/llama_tokenized-global-chunks/fineweb-edu/fineweb-edu/",
    "s3://smollm3/datasets/llama_tokenized-global-chunks/dclm/dclm/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/pes2o/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/wiki/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stackexchange/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-fra/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-spa/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-deu/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-ita/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-por/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-cmn/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-rus/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-fas/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-jpn/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-kor/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-hin/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-tha/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-vie/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/fw2-ell/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/infiwebmath/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/finemath/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-Python/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-Java/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-JavaScript/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-C/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-Cpp/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-C-Sharp/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-PHP/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-TypeScript/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-Swift/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-SQL/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-Ruby/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-Markdown/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-HTML/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-Rust/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-Go/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/stack-edu-Shell/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/pull-requests/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/kaggle/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/jupyter-scripts/",
    "s3://smollm3/datasets/llama_tokenized-individual-chunks/github-issues/"
]

# Mixing weights
dataset_weights = [
    0.333, 0.37, 0.02, 0.001, 0.004, 0.016, 0.02, 0.022, 0.0105, 0.01,
    0.01, 0.01, 0.003, 0.00325, 0.00325, 0.00325, 0.00325, 0.00325, 0.00225, 0.01,
    0.017, 0.025, 0.013, 0.013, 0.007, 0.018, 0.006, 0.006, 0.003, 0.001,
    0.004, 0.0008, 0.005, 0.006, 0.0008, 0.0005, 0.0007, 0.006, 0.0005, 0.0055,
    0.0032
]

# Configuration
TOTAL_TOKENS_NEEDED = 2.6e12  # 2.6 trillion tokens
TOKENS_PER_SHARD = 100e9      # 100 billion tokens per shard

def format_size(size_bytes):
    """Format size in bytes to human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"

def parse_size(size_str):
    """Parse size string like '372.5G' to bytes"""
    if size_str.endswith('G'):
        return float(size_str[:-1]) * 1024**3
    elif size_str.endswith('M'):
        return float(size_str[:-1]) * 1024**2
    elif size_str.endswith('K'):
        return float(size_str[:-1]) * 1024
    else:
        return float(size_str)

def get_metadata_token_count(metadata_path, dry_run=True):
    """Get token count from .ds.metadata file"""
    if dry_run:
        return None
    
    try:
        # Download metadata file content to stdout
        result = subprocess.run(['s5cmd', 'cat', metadata_path], 
                              capture_output=True, text=True, check=True)
        
        # The second line contains the token count
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            return int(lines[1])
        else:
            print(f"⚠️  Metadata file {metadata_path} has unexpected format")
            return None
            
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"⚠️  Error reading metadata from {metadata_path}: {e}")
        return None

def list_s3_files(s3_path, dry_run=True):
    """List .ds files from S3 path using s5cmd and get actual token counts"""
    if dry_run:
        print(f"[DRY RUN] Would list files from: {s3_path}")
        return []
    
    try:
        result = subprocess.run(['s5cmd', 'ls', s3_path], 
                              capture_output=True, text=True, check=True)
        
        # First, collect all .ds files and their corresponding .metadata files
        ds_files = {}
        metadata_files = {}
        
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 4:
                size_str = parts[2]
                filepath = parts[3]
                filename = os.path.basename(filepath)
                
                if filename.endswith('.ds') and not filename.endswith('.ds.metadata'):
                    ds_files[filename] = {
                        'filename': filename,
                        'path': filepath if filepath.startswith('s3://') else f"{s3_path.rstrip('/')}/{filename}",
                        'size_bytes': parse_size(size_str),
                        'size_str': size_str
                    }
                elif filename.endswith('.ds.metadata'):
                    metadata_files[filename] = filepath if filepath.startswith('s3://') else f"{s3_path.rstrip('/')}/{filename}"
        
        # Match .ds files with their .metadata files and get token counts
        files = []
        print(f"   Found {len(ds_files)} .ds files, reading token counts...")
        
        for ds_filename, ds_info in ds_files.items():
            metadata_filename = ds_filename + '.metadata'
            
            if metadata_filename in metadata_files:
                metadata_path = metadata_files[metadata_filename]
                token_count = get_metadata_token_count(metadata_path, dry_run=False)
                
                if token_count is not None:
                    ds_info['token_count'] = token_count
                    print(f"   📊 {ds_filename}: {token_count:,} tokens")
                else:
                    # Fallback to estimation if metadata can't be read
                    ds_info['token_count'] = int(TOKENS_PER_SHARD)  # Use as fallback
                    print(f"   📊 {ds_filename}: {int(TOKENS_PER_SHARD):,} tokens (estimated)")
            else:
                print(f"   ⚠️  No metadata file found for {ds_filename}, using estimate")
                ds_info['token_count'] = int(TOKENS_PER_SHARD)  # Use as fallback
            
            files.append(ds_info)
        
        # Sort files by filename for consistent ordering
        files.sort(key=lambda x: x['filename'])
        return files
        
    except subprocess.CalledProcessError as e:
        print(f"Error listing {s3_path}: {e}")
        return []

def calculate_files_needed(files, tokens_needed):
    """Calculate which files to download based on actual token counts"""
    if not files:
        return []
    
    selected_files = []
    accumulated_tokens = 0
    
    for file_info in files:
        if accumulated_tokens >= tokens_needed:
            break
            
        # Use actual token count from metadata
        actual_tokens = file_info.get('token_count', int(TOKENS_PER_SHARD))
        selected_files.append({
            **file_info,
            'actual_tokens': actual_tokens
        })
        accumulated_tokens += actual_tokens
    
    return selected_files

def calculate_shards_needed(dry_run=True):
    """Calculate the files needed for each dataset."""
    
    print(f"Total tokens needed: {TOTAL_TOKENS_NEEDED:,.0f}")
    print(f"Estimated tokens per 100GB shard: {TOKENS_PER_SHARD:,.0f} (fallback only)")
    print("=" * 100)
    
    results = []
    total_tokens_allocated = 0
    total_files = 0
    total_size_bytes = 0
    
    for i, (dataset, weight) in enumerate(zip(datasets, dataset_weights)):
        # Calculate tokens needed for this dataset
        tokens_needed = TOTAL_TOKENS_NEEDED * weight
        
        # Extract dataset name from path
        dataset_name = dataset.split('/')[-2] if dataset.endswith('/') else dataset.split('/')[-1]
        
        print(f"\n📁 Processing {dataset_name} (weight: {weight:.4f})")
        print(f"   Tokens needed: {tokens_needed:,.0f}")
        
        # List files from S3
        files = list_s3_files(dataset, dry_run=dry_run)
        
        if not files and not dry_run:
            print(f"   ⚠️  No .ds files found in {dataset}")
            continue
        
        if dry_run:
            # For dry run, estimate based on the pattern you showed
            estimated_files_needed = max(1, math.ceil(tokens_needed / TOKENS_PER_SHARD))
            # Estimate size based on 372.5GB per file (from your example)
            estimated_size_per_file = 372.5 * (1024**3)  # 372.5GB in bytes
            estimated_total_size = estimated_files_needed * estimated_size_per_file
            
            print(f"   📦 Estimated files needed: {estimated_files_needed}")
            print(f"   💾 Estimated download size: {format_size(estimated_total_size)}")
            
            result = {
                'dataset': dataset_name,
                'path': dataset,
                'weight': weight,
                'tokens_needed': tokens_needed,
                'files_needed': estimated_files_needed,
                'actual_tokens': estimated_files_needed * TOKENS_PER_SHARD,
                'total_size_bytes': estimated_total_size,
                'files': []
            }
        else:
            # Calculate actual files needed
            selected_files = calculate_files_needed(files, tokens_needed)
            actual_tokens = sum(f.get('actual_tokens', 0) for f in selected_files)
            total_size_bytes = sum(f.get('size_bytes', 0) for f in selected_files)
            
            print(f"   📦 Files available: {len(files)}")
            print(f"   📦 Files to download: {len(selected_files)}")
            print(f"   💾 Total download size: {format_size(total_size_bytes)}")
            if selected_files:
                total_actual_tokens = sum(f.get('token_count', 0) for f in files)
                avg_tokens_per_file = total_actual_tokens / len(files) if files else 0
                print(f"   📊 Average tokens per file: {avg_tokens_per_file:,.0f}")
            
            result = {
                'dataset': dataset_name,
                'path': dataset,
                'weight': weight,
                'tokens_needed': tokens_needed,
                'files_needed': len(selected_files),
                'actual_tokens': actual_tokens,
                'total_size_bytes': total_size_bytes,
                'files': selected_files
            }
        
        results.append(result)
        total_tokens_allocated += result['actual_tokens']
        total_files += result['files_needed']
        total_size_bytes += result.get('total_size_bytes', 0)
        
        excess = result['actual_tokens'] - tokens_needed
        print(f"   ✅ Files: {result['files_needed']:3d} | "
              f"Size: {format_size(result.get('total_size_bytes', 0)):>8s} | "
              f"Actual tokens: {result['actual_tokens']:12,.0f} | "
              f"Excess: {excess:+12,.0f}")
    
    print("\n" + "=" * 100)
    print(f"📊 SUMMARY")
    print("=" * 100)
    print(f"Total files to download: {total_files}")
    print(f"Total download size: {format_size(total_size_bytes)}")
    print(f"Total tokens that will be downloaded: {total_tokens_allocated:,.0f}")
    print(f"Excess tokens (overhead): {total_tokens_allocated - TOTAL_TOKENS_NEEDED:,.0f}")
    if TOTAL_TOKENS_NEEDED > 0:
        print(f"Overhead percentage: {((total_tokens_allocated - TOTAL_TOKENS_NEEDED) / TOTAL_TOKENS_NEEDED) * 100:.2f}%")
    
    return results

def generate_download_scripts(results, dry_run=True):
    """Generate s5cmd download scripts for each dataset."""
    
    print("\n" + "=" * 100)
    print("S5CMD DOWNLOAD SCRIPTS")
    print("=" * 100)

    partition = "lrd_all_serial" # "boost_usr_prod"
    proxy = ""

    # Create individual download scripts for each dataset
    for result in results:
        if result['files_needed'] == 0:
            continue
            
        dataset_name = result['dataset']
        script_name = f"download_{dataset_name}.sh"
        
        print(f"\n🔧 Generated script: {script_name}")

        # NOTE: (MC) this is mainly for Leonardo cluster and parallel downloads as I had 10TB space, but can be adapted
        if partition == "lrd_all_serial":
            slurm_cmd = f"""
#SBATCH --job-name=download_sl3-{dataset_name}
#SBATCH --time=4:00:00
#SBATCH --partition=lrd_all_serial
#SBATCH --output={dataset_name}/log_%j.out
#SBATCH --error={dataset_name}/log_%j.err
        """
        else:
            account = "ACCOUNT_TO_FILL"
            qos = "normal"
            time = "12:00:00"
            slurm_cmd = f"""
#SBATCH --job-name=download_sl3-{dataset_name}
#SBATCH --time={time}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --qos={qos}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --output={dataset_name}/log_%j.out
#SBATCH --error={dataset_name}/log_%j.err
        """
            proxy = """
# Internet proxy for Leonardo 
full_hostname=$(hostname)
if [[ "$full_hostname" == *"lrdn"* ]]; then
    port=$(cat $HOME/scripts/.leonardo_port)
    echo Waiting the reverse proxy...

    while ! netstat -an | grep $port &> /dev/null; do sleep 1; done
    export HTTP_PROXY=socks5://127.0.0.1:$port
    export HTTPS_PROXY=socks5://127.0.0.1:$port
    export SOCK_PROXY=socks5://127.0.0.1:$port
    export ALL_PROXY=socks5://127.0.0.1:$port
    echo Reverse proxy is up and running!
fi
""" 
        
        script_content = f"""#!/bin/bash
{slurm_cmd}

# Download script for {dataset_name}
# Tokens needed: {result['tokens_needed']:,.0f}
# Files to download: {result['files_needed']}
# Estimated download size: {format_size(result.get('total_size_bytes', 0))}

{proxy}

S3_BASE="{result['path'].rstrip('/')}"
LOCAL_BASE="{dataset_name}"

# Create local directory
mkdir -p "$LOCAL_BASE"

# List files and download only .ds files (excluding .metadata)
echo "📥 Listing .ds files from S3..."
s5cmd ls "$S3_BASE/" | grep '\\.ds$' | head -{result['files_needed']} > filelist_{dataset_name}.txt

echo "🚀 Starting sequential downloads for {dataset_name}..."
while read -r line; do

    # Parse s5cmd ls output: date time size filename
    size=$(echo "$line" | awk '{{print $3}}')
    path_fragment=$(echo "$line" | awk '{{print $4}}')
    filename=$(basename "$path_fragment")
    filename_local="$LOCAL_BASE/$filename"
    
    # Skip if line is malformed
    if [[ -z "$size" || -z "$path_fragment" ]]; then
        continue
    fi
    
    # Ensure full S3 path
    if [[ "$path_fragment" == s3://* ]]; then
        s3path="$path_fragment"
    else
        s3path="${{S3_BASE}}/${{filename}}"
    fi
    
    # Check if file already exists and matches expected size
    if [[ -f "$filename_local" ]]; then
        local_size=$(stat -c %s "$filename_local" 2>/dev/null || echo "0")
        if [[ "$local_size" -eq "$size" ]]; then
            echo "✅ $filename already downloaded, skipping"
            continue
        else
            echo "⚠️  $filename exists but size mismatch, re-downloading"
            rm -f "$filename_local"
        fi
    fi
    
    # Download file
    echo "⬇️  Downloading $filename (size: $size)"
    s5cmd cp --show-progress --part-size=500 "$s3path" "$LOCAL_BASE/"
    
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to download $filename"
        exit 1
    fi
done < filelist_{dataset_name}.txt

echo "✅ {dataset_name} downloads completed."
"""
        
        if not dry_run:
            with open(script_name, 'w') as f:
                f.write(script_content)
            os.chmod(script_name, 0o755)
            print(f"   💾 Saved executable script: {script_name}")
        else:
            print("   [DRY RUN] Script content preview:")
            print("   " + "\n   ".join(script_content.split('\n')[:15]) + "\n   ...")

def generate_storage_report(results):
    """Generate a detailed storage space report"""
    print("\n" + "=" * 100)
    print("💾 STORAGE SPACE REPORT")
    print("=" * 100)
    
    # Sort by size for better overview
    sorted_results = sorted(results, key=lambda x: x.get('total_size_bytes', 0), reverse=True)
    
    total_size = sum(r.get('total_size_bytes', 0) for r in results)
    
    print(f"{'Dataset':<25} {'Files':<6} {'Size':<10} {'% of Total':<10} {'Tokens':<15}")
    print("-" * 80)
    
    for result in sorted_results:
        if result['files_needed'] > 0:
            size_bytes = result.get('total_size_bytes', 0)
            size_str = format_size(size_bytes)
            percent = (size_bytes / total_size * 100) if total_size > 0 else 0
            
            print(f"{result['dataset']:<25} {result['files_needed']:<6} {size_str:<10} "
                  f"{percent:>8.1f}% {result['actual_tokens']:>13,.0f}")
    
    print("-" * 80)
    print(f"{'TOTAL':<25} {sum(r['files_needed'] for r in results):<6} {format_size(total_size):<10} "
          f"{'100.0%':<10} {sum(r['actual_tokens'] for r in results):>13,.0f}")
    
    # Storage recommendations
    print(f"\n💡 STORAGE RECOMMENDATIONS:")
    recommended_free = total_size * 1.2  # 20% extra space
    print(f"   • Ensure at least {format_size(recommended_free)} of free disk space")
    print(f"   • Consider using fast SSDs for the largest datasets:")
    
    large_datasets = [r for r in sorted_results if r.get('total_size_bytes', 0) > total_size * 0.1]
    for dataset in large_datasets[:3]:  # Top 3 largest
        print(f"     - {dataset['dataset']}: {format_size(dataset.get('total_size_bytes', 0))}")

def generate_master_script(results, dry_run=True):
    """Generate a master script to download all datasets."""
    
    total_size = sum(r.get('total_size_bytes', 0) for r in results if r['files_needed'] > 0)
    
    script_content = f"""#!/bin/bash
# Master download script for all datasets
# Total estimated download size: {format_size(total_size)}
# Generated automatically

set -e  # Exit on any error

DATASETS=("""
    
    for result in results:
        if result['files_needed'] > 0:
            script_content += f'    "{result["dataset"]}"\n'
    
    script_content += """)

echo "🚀 Starting download of all datasets..."
echo "Total datasets: ${#DATASETS[@]}"
echo "📊 Total estimated download size: """ + format_size(total_size) + """"

# Check available disk space
REQUIRED_SPACE=""" + str(int(total_size * 1.2)) + """  # 20% extra space
AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
AVAILABLE_BYTES=$((AVAILABLE_SPACE * 1024))

if [ "$AVAILABLE_BYTES" -lt "$REQUIRED_SPACE" ]; then
    echo "⚠️  WARNING: Insufficient disk space!"
    echo "   Required: """ + format_size(int(total_size * 1.2)) + """ (with 20% buffer)"
    echo "   Available: $(df -h . | tail -1 | awk '{print $4}')B"
    echo "   Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ Aborted by user"
        exit 1
    fi
fi

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=" * 80
    echo "📦 Starting download for: $dataset"
    echo "=" * 80
    
    if [[ -f "download_${dataset}.sh" ]]; then
        ./download_${dataset}.sh
        if [[ $? -eq 0 ]]; then
            echo "✅ $dataset completed successfully"
        else
            echo "❌ $dataset failed"
            exit 1
        fi
    else
        echo "⚠️  Script download_${dataset}.sh not found, skipping"
    fi
done

echo ""
echo "🎉 All downloads completed successfully!"
"""
    
    if not dry_run:
        with open("download_all.sh", 'w') as f:
            f.write(script_content)
        os.chmod("download_all.sh", 0o755)
        print("\n💾 Saved master script: download_all.sh")
    else:
        print("\n🔧 Generated master script: download_all.sh")
        print("[DRY RUN] Use --execute flag to save actual scripts")

if __name__ == "__main__":
    import sys
    
    # Check for --execute flag
    execute = "--execute" in sys.argv
    
    if not execute:
        print("🔍 DRY RUN MODE - No actual S3 calls or files will be written")
        print("Use --execute flag to run actual s5cmd operations and generate scripts")
        print()
    
    # Verify weights sum to 1
    total_weight = sum(dataset_weights)
    print(f"Total weight sum: {total_weight:.6f}")
    if abs(total_weight - 1.0) > 1e-6:
        print(f"WARNING: Weights don't sum to 1.0! Difference: {total_weight - 1.0:.6f}")
    print()
    
    # Calculate files needed
    results = calculate_shards_needed(dry_run=not execute)
    
    # Generate download scripts
    generate_download_scripts(results, dry_run=not execute)
    generate_master_script(results, dry_run=not execute)
    generate_storage_report(results)
    
    if not execute:
        print("\n" + "=" * 100)
        print("💡 NEXT STEPS:")
        print("1. Run with --execute flag to generate actual download scripts")
        print("2. Review generated scripts before running")
        print("3. Execute ./download_all.sh or individual dataset scripts")
        print("4. Monitor downloads and disk space")
