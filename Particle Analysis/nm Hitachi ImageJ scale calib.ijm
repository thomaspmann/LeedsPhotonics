path = getDirectory("image");
name = getTitle();
texname = replace(name,".tif",".txt");
run("Measure")

     lineseparator = "\n\=";
     cellseparator = "\t\";
     lines=split(File.openAsString(path+texname), lineseparator);
     labels=split(lines[0], cellseparator);
     if (labels[0]==" ")
        k=1;
     else
        k=0; 
     for (j=k; j<labels.length; j++)
        setResult(labels[j],0,0);

     run("Clear Results");
     for (i=1; i<lines.length; i++) {
        items=split(lines[i], cellseparator);
        for (j=k; j<items.length; j++)
           setResult(labels[j],i-1,items[j]);
}
     
size = getResultString("[SemImageFile]",(26));
run("Set Scale...", "distance=1 known=size pixel=1 unit=nm");
selectWindow("Results"); 
   run("Close"); 
   run("Clear Results");
   run("Save");
   