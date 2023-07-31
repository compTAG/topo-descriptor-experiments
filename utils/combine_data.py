import os
import csv
import numpy as np
import copy
import sys
from PIL import Image
from PyPDF2 import PdfFileMerger, PdfFileReader


def find_stats(in_file,exp):
  with open(in_file) as f:
    stats = csv.DictReader(f)
    stats_row = next(stats)
    if exp == "smallest_angle_exp":
      return {"n": stats_row['n'],
          "minSize":stats_row['min_angle'],
          "fineStratum": stats_row['num_stratum'],
          "necessaryStratum": stats_row['num_needed_stratum']
          }
    else:
      for row in stats:
      # grab the right row and trim out data that has too many stratum
        if int(row['samples'])==16384 and int(row['num_stratum']) <= 5000:
          return {"n": row['n'],
            "samples":row['samples'],
            "hits": row['hits'],
            "num_stratum": row['num_stratum']
            }
      return -1

def write_stats(error_stats,out_file, exp):
  if exp == "smallest_angle_exp":
    with open(out_file, 'w') as f:
            f.write("n,minSize,fineStratum,necessaryStratum\n")
            for graph in error_stats:
                    f.write(str(graph['n'])+","+
                                    str(graph['minSize'])+","+
                                    str(graph['fineStratum'])+","+
                                    str(graph['necessaryStratum'])+"\n")
  else: 
    with open(out_file, 'w') as f:
            f.write("n,samples,hits,num_stratum\n")
            for graph in error_stats:
                    f.write(str(graph['n'])+","+
                                    str(graph['samples'])+","+
                                    str(graph['hits'])+","+
                                    str(graph['num_stratum'])+"\n")


def rando(exp, approx, file_name):
  if exp == "smallest_angle_exp":
    num_points = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for k in num_points:
      error_stats = []
      out_file = os.path.join("analysis_" + approx + "_approx", exp, "combined_data", "random", file_name + "_" + str(k)+".txt")
      for i in range(0,100):
        in_file = os.path.join('output_' + approx + '_approx', exp, 'random', 'RAND_'+str(k)+"_"+str(i)+".txt")
        error_stats.append(find_stats(in_file,exp))
      write_stats(error_stats, out_file, exp)
  else:
    num_points = [3, 5, 10, 20, 30, 40, 50, 60, 70]
    for k in num_points:
      error_stats = []
      out_file = os.path.join("analysis_" + approx + "_approx", exp, "combined_data", "random", file_name + "_" + str(k)+".txt")
      for i in range(0,100):
              in_file = os.path.join('output_' + approx + '_approx', exp, 'random', 'RAND_'+str(k)+"_"+str(i)+".txt")
              stats = find_stats(in_file,exp)
              if stats != -1:
                    error_stats.append(stats)
    # print(len(error_stats))
      write_stats(error_stats, out_file,exp)


def mpeg7_mnist(data_type, exp, approx,file_name):
  if exp == "smallest_angle_exp":
    error_stats = []
    out_file = os.path.join('analysis_' + approx + '_approx', exp, 'combined_data', data_type, file_name + '.txt')
    for filename in os.listdir(os.path.join('output_' + approx + '_approx', exp, data_type)):
      in_file = os.path.join('output_' + approx + '_approx', exp, data_type, filename)
      error_stats.append(find_stats(in_file,exp))
    write_stats(error_stats, out_file, exp)
  else:
    error_stats = []
    out_file = os.path.join('analysis_' + approx + '_approx', exp, 'combined_data', data_type, file_name + '.txt')
    for filename in os.listdir(os.path.join('output_' + approx + '_approx', exp, data_type)):
      in_file = os.path.join('output_' + approx + '_approx', exp, data_type, filename)
      stats = find_stats(in_file,exp)
      if stats != -1:
        error_stats.append(stats)
      else:
        print("Error")
        print(in_file)
    # print(len(error_stats))
    write_stats(error_stats, out_file,exp)

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

  #Convert and save a pdf version
  image = Image.open(os.path.join('figs', exp, exp + '_' + approx +'.png'))
  img_pdf = image.convert('RGB')
  img_pdf.save(os.path.join('figs', exp, exp + '_' + approx +'.pdf'))


def combine_pdfs(exp,approx):
  file_names = [os.path.join('figs',exp,'random', 'random_' + exp +'.pdf'), 
                                  os.path.join('figs', exp, 'mnist', 'mnist_' +  approx + '_approx_' + exp + '.pdf'), 
                                  os.path.join('figs', exp, 'mpeg7', 'mpeg7_' + approx + '_approx_' + exp + '.pdf')]
  merger = PdfFileMerger()
  for file in file_names:
      merger.append(file)

  merger.write(os.path.join('figs', exp, exp + '_' + approx +'.pdf'))                              



