echo "Combining parts back into .zip file"
cat re.?? > random_embeddings.zip
echo "Unzipping .zip file"
unzip random_embeddings.zip
echo "Cleaning up .zip file"
rm random_embeddings.zip
echo "Moving .npy datafile into place"
mv ./random_embeddings.npy ./../random_embeddings.npy
