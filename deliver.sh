# ZIPs the current directory without the .git folder

# Clean git ignored files
git clean -dfx

# Get the current directory name
DIR=${PWD##*/}

# Create the zip file name
ZIP_FILE_NAME="$DIR.zip"

# Zip the current directory (including the directory itself)
cd ..
zip -r "$DIR/$ZIP_FILE_NAME" "$DIR" -x "$DIR/*.git*" -x "$DIR/*.DS_Store"
