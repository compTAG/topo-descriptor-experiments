import os
import csv
import numpy as np
import copy
import sys
from PIL import Image

def combine_pngs(exp,approx):
  images = [Image.open(x) for x in [os.path.join('figs',exp,'random', 'random_' + exp +'.png'), 
                                  os.path.join('figs', exp, 'mnist', 'mnist_' +  approx + '_approx_' + exp + '.png'), 
                                  os.path.join('figs', exp, 'mpeg7', 'mpeg7_' + approx + '_approx_' + exp + '.png')]]
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save(os.path.join('figs', exp, exp + '_' + approx +'.png'))


def find_stats(in_file):
  with open(in_file) as f:
    stats = csv.DictReader(f)
    stats_row = next(stats)
    return {"n": stats_row['n'],
        "minSize":stats_row['min_angle'],
        "fineStratum": stats_row['num_stratum'],
        "necessaryStratum": stats_row['num_needed_stratum']
        }

def write_stats(error_stats,out_file):
  with open(out_file, 'w') as f:
    f.write("n,minSize,fineStratum,necessaryStratum\n")
    for graph in error_stats:
      f.write(str(graph['n'])+","+
          str(graph['minSize'])+","+
          str(graph['fineStratum'])+","+
          str(graph['necessaryStratum'])+"\n")


def random(exp, approx, file_name):
  num_points = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
  for k in num_points:
    error_stats = []
    out_file = os.path.join("analysis_" + approx + "_approx", exp, "combined_data", "random", file_name + "_" + str(k)+".txt")
    for i in range(0,100):
      in_file = os.path.join('output_' + approx + '_approx', exp, 'random', 'RAND_'+str(k)+"_"+str(i)+".txt")
      error_stats.append(find_stats(in_file))
    write_stats(error_stats, out_file)

def mpeg7_mnist(data_type, exp, approx,file_name):
  error_stats = []
  out_file = os.path.join('analysis_' + approx + '_approx', exp, 'combined_data', data_type, file_name + '.txt')
  for filename in os.listdir(os.path.join('output_' + approx + '_approx', exp, data_type)):
    in_file = os.path.join('output_' + approx + '_approx', exp, data_type, filename)
    error_stats.append(find_stats(in_file))
  write_stats(error_stats, out_file)

#def main():
  #random("smallest_angle_exp", "001", 'angle_stats')
  #mpeg7_mnist("mpeg7", "smallest_angle_exp", "001", 'angle_stats')
  #mpeg7_mnist("mnist","smallest_angle_exp", "001", 'angle_stats')

#if __name__ == '__main__':main()
