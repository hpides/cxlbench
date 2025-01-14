find . -name "*.pdf" -print0 | xargs -0 -I {} sh -c 'output="${0%.pdf}.png"; pdftoppm -png "{}" "${output%.png}"' {}
