#!/bin/bash

# Script to clean up large files from git history
# WARNING: This will rewrite git history!

echo "This script will clean large files from git history."
echo "Make sure you have a backup before proceeding!"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "Removing large files from git history..."

# Remove large data files from history
git filter-branch --force --index-filter \
  'git rm -r --cached --ignore-unmatch data/*.tar.gz data/ffhq data/ffhq2 data/*.xlsx' \
  --prune-empty --tag-name-filter cat -- --all

echo "Cleaning up..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Repository cleaned!"
echo "New .git size:"
du -sh .git

echo ""
echo "To push these changes to GitHub (WARNING: This will rewrite history!):"
echo "git push origin --force --all"
echo "git push origin --force --tags"