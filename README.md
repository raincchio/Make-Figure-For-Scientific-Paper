# Make-Figure-For-Scientific-Paper
We built this repository hoping to provide some help to those who are not familiar with drawing high-quality paper graphs.

## Data Relative Path Structure
The relative path is organized as "./task/algo/domain/seed/progress.csv."

The design intuition of this path structure is that we want to collect data for one task and draw figure for single task. The second layer path is designed as “algo” rather than “domain” because we usually keep adjusting our algorithm. We may need to switch directory to check the algorithm results  if the second layer is “domain.” The seed directory and the result file name are stripped, and the seed is not used to name the result for better scalability.

In each "progress.csv" file, the result is organized like the following:

+ 1st row: epoch, metric1, metric2

+ 2st row: 0, result1, result2
+ ....

## Figure Size
+ One column width: 3.5 inches, 88.9 millimeters, or 21 picas 
+ Two columns width: 7.16 inches, 182 millimeters, or 43 picas

Generally speaking, you should only consider the width of your figure. If you want to put three figures in one row, the width should be less than 7.16/3 inches, and the height may be 0.618 * width or 0.75 * width to look comfortable.

## Plot Figure

### Line Chart [Final]


![plot](./png/linechart.png)

### 3D Line Chart [Ongoing]
We are trying to make a nice 3D line chart to show multiple domain results.

![plot](./png/3dlinechart.png)

