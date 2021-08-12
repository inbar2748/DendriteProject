# SOA

## Install
The `requirements.txt` file list all requirements Python libraries , and they will be installed using:

```
pip install -r requirements.txt
```

```
python ./SOA.py
```


##  PROTOCOL: 

The Israeli Ministry of Health approved the use of mice under protocol IL-218-01-21 for the ethical use of experimental animals.
SOA is designed to work with Windows 10 only and Python 3. 
Additional examples of analysis performed using the software can be seen here as well. 



1.	Open the SOA application.

1.1.	Open URL address: https://mega.nz/folder/bKZhmY4I#4WAaec4biiGt4_1lJlL4WA, find the SOA.zip zipped folder and download the ZIP file by double-click.

1.2.	Unzip the folder - right-click on SOA.zip and choose Extract Files. The Extraction Path and Options window opens, and the Destination Address text box displays the path for the extracted files. To extract to a different location, click one of the folders in the window's right panel to make it the destination folder. Click OK to extract the files to that folder. 

1.3.	Open the extract SOA file and double-click on SOA.exe. A black window will open and after that the application will show.


2.	Open an image to analyze.

2.1.	In the SOA Viewer Upload menu bar select Choose File > choose an image from the computer files and click it (NOTE: .png .jpg .tif .bmp files only) >Open > path of the file will show > Next. 

3.	Segmentation optimization. 

NOTE: In the SOA Viewer Properties menu bar, you can change various parameters in the segmentation process by changing the values in the selected parameter. 

3.1.	In Edges, Adjust the threshold for the display by selecting the Threshold and entering a number, the lower the number, the more lines are detected. Threshold is a number that ranges from 0 to 255. The default value has been set to 0. 

3.2.	In Merge Lines:

3.2.1.	Adjust the minimum distance to merge for the display by selecting the Min distance to merge and entering a number. 

NOTE: The Min distance to merge ranges from 0 to 30 pixels. The default value is set to 20. 

3.2.2.	Adjust the minimum angle to merge for the display by selecting the Min angle to merge and entering a number. 

NOTE: The Min angle to merge ranges from 0 to 30 degrees. The default value is set to 10.

3.3.	Click on Create Preview Segmentation Image. 

NOTE: A preview image of the segmentation results will be displayed according to the updated values. Also, the number of lines before merging and the number of lines after merging will be displayed. 

3.4.	The parameters can be changed by the user to reach maximum identification of segments. If there is a need to change the Properties, click on the close window button > follow the steps in the manual tracing directions in sections 3.1-3.4.

4.	Need a heading here

4.1.	Create the segmentation files and analysis result files.
Press > Ok to visualize the segmentation images and the analyzing graphs. A window will appear for selecting a location where the .xlsx file will be saved > insert a file name > Choose Save > the .xlsx file with data information will be created and saved. In addition to the .xlsx file, the following files will be automatically displayed: A file that presents the original image, the line recognition image, and the final image of the segmentation and three analysis graphs.

5.	Navigation toolbar

NOTE: A navigation toolbar is included in all figure windows and can be used to navigate through the data set. Each of the buttons at the bottom of the toolbar is described below.

5.1.	To navigate back and forth between previously defined views, use the Forward and Back buttons. 

NOTE: The Home, Forward, and Back buttons are similar to the Home, Forward, and Back controls on a web browser. Home always takes you to the first, default view of your data. 

5.2.	Use the Zoom button to pan and zoom. To activate panning and zooming press the Zoom button, then move your mouse to a desired location in the image.  

5.2.1.	To pan the figure, press and hold the left mouse button while dragging it to a new position. When you let go, the data beneath the point where you pressed will be moved to the point where you let go. If you hold down the 'x' or 'y' keys while panning, the motion will be limited to the x or y axes, respectively. 

5.2.2.	To zoom, hold down the right mouse button and drag it to a new location. The x axis will zoom in proportion to rightward movement and out proportionately to leftward movement. The same is true for the y axis and up/down motions. When you begin zooming, the point under your mouse remains stationary, allowing you to zoom in or out as much as you want around that point. The modifier keys 'x', 'y', or 'CONTROL' can be used to limit the zoom to the x, y, or aspect ratio preserve, respectively.

5.3.	The Zoom-to-rectangle button. To activate this mode, click this Zoom-to-rectangle button. Place your cursor over the image and press the left mouse button. Drag the mouse to a new location while holding the button to define a rectangular region. The axes view limits will be zoomed to the defined region when the left mouse button is pressed. The axes view limits will be zoomed out when you use the right mouse button, placing the original axes in the defined region.

5.4.	The Subplot-configuration button. Use this tool to configure the appearance of the subplot. You can stretch or compress the subplot's left, right, top, or bottom sides, as well as the space between rows and columns.

5.5.	The Save button. To open a file save dialog, click this button. The file can be saved in the following formats: .png, .ps, .eps, .svg, or .pdf.
