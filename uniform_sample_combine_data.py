import os
import csv
import numpy as np
import copy
import sys
from PIL import Image

def combine_pngs(exp):
  images = [Image.open(x) for x in ['figs/' + exp +'/random/random_' + exp +'.png', 
                                  'figs/' + exp + '/mnist/mnist_001_approx_' + exp + '.png', 
                                  'figs/' + exp + '/mpeg7/mpeg7_001_approx_' + exp + '.png']]
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('figs/' + exp + '/' + exp +'_001.png')

def find_stats(in_file):
  with open(in_file) as f:
    stats = csv.DictReader(f)
    #we want the 12th row...
    # stats_row = next(stats)
    for row in stats:
      # grab the right row and trim out data that has too many stratum
      if int(row['samples'])==16384 and int(row['num_stratum']) <= 5000:
        return {"n": row['n'],
          "samples":row['samples'],
          "hits": row['hits'],
          "num_stratum": row['num_stratum']
          }
    return -1

def write_stats(error_stats,out_file):
  with open(out_file, 'w') as f:
    f.write("n,samples,hits,num_stratum\n")
    for graph in error_stats:
      f.write(str(graph['n'])+","+
          str(graph['samples'])+","+
          str(graph['hits'])+","+
          str(graph['num_stratum'])+"\n")

def random(exp, approx):
  num_points = [3, 5, 10, 20, 30, 40, 50, 60, 70]
  for k in num_points:
    error_stats = []
    out_file = "analysis_" + approx + "_approx/" + exp + "/combined_data/random/sample_stats_"+str(k)+".txt"
    for i in range(0,100):
      in_file = 'output_' + approx + '_approx/' + exp + '/random/RAND_'+str(k)+"_"+str(i)+".txt"
      stats = find_stats(in_file)
      if stats != -1:
        error_stats.append(stats)
    # print(len(error_stats))
    write_stats(error_stats, out_file)

def mpeg7_mnist(data_type, exp, approx):
  error_stats = []
  out_file = 'analysis_' + approx + '_approx/' + exp + '/combined_data/'+data_type+'/sample_stats.txt'
  for filename in os.listdir('output_' + approx + '_approx/' +exp + '/'+data_type):
    in_file = 'output_' + approx + '_approx/' + exp + '/' + data_type+"/"+filename
    stats = find_stats(in_file)
    if stats != -1:
      error_stats.append(stats)
    else:
      print "Error"
      print(in_file)
  # print(len(error_stats))
  write_stats(error_stats, out_file)

def main():
  random("uniform_sample_exp", "001")
  mpeg7_mnist("mpeg7","uniform_sample_exp", "001")
  mpeg7_mnist("mnist","uniform_sample_exp", "001")

if __name__ == '__main__':main()