#!/bin/bash

# Script to completely remove data folder from all git history
# This will rewrite git history permanently!

echo "==============================================="
echo "WARNING: This will PERMANENTLY remove the 'data' folder"
echo "from ALL git history and rewrite your repository!"
echo "==============================================="
echo ""
echo "Current repository size:"
du -sh .git
echo ""

# Use git filter-repo if available (preferred method)
if command -v git-filter-repo &> /dev/null; then
    echo "Using git-filter-repo (recommended method)..."
    git filter-repo --path data/ --invert-paths --force
else
    echo "Using BFG Repo-Cleaner alternative method..."
    echo "Removing 'data' folder from all commits..."

    # Using git filter-branch (older method, but works)
    git filter-branch --force --index-filter \
      'git rm -rf --cached --ignore-unmatch data/' \
      --prune-empty --tag-name-filter cat -- --all

    echo "Cleaning up refs and garbage collection..."

    # Clean up original refs
    rm -rf .git/refs/original/

    # Expire all reflog entries
    git reflog expire --expire=now --all

    # Garbage collect aggressively
    git gc --prune=now --aggressive
fi

echo ""
echo "==============================================="
echo "Repository cleaned!"
echo "New .git size:"
du -sh .git
echo ""
echo "New repository total size:"
du -sh .
echo "==============================================="
echo ""
echo "IMPORTANT NEXT STEPS:"
echo "1. The data folder is now completely removed from history"
echo "2. To push these changes to GitHub (This WILL rewrite history!):"
echo "   git push origin --force --all"
echo "   git push origin --force --tags"
echo ""
echo "3. All collaborators will need to re-clone the repository"
echo "   or rebase their local branches"
echo ""
echo "4. The 'data' folder still exists locally but is not in git."
echo "   You may want to back it up elsewhere if needed."