#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run
# > python flickr_GetUrl.py tag number_of_images_to_attempt_to_download

from flickrapi import FlickrAPI
import pandas as pd
import sys
import csv
import requests
import os
import time

key = ''
secret = ''


def put_images(file_name):
    urls = []
    with open(file_name, newline="") as csvfile:
        doc = csv.reader(csvfile, delimiter=",")
        for row in doc:
            if row[1].startswith("https"):
                urls.append(row[1])
    if not os.path.isdir(os.path.join(os.getcwd(), file_name.split("_")[0])):
        os.mkdir(file_name.split("_")[0])
    t0 = time.time()
    for url in enumerate(urls):
        print("Starting download {} of ".format(url[0]+1), len(urls))
        try:
            resp = requests.get(url[1], stream=True)
            path_to_write = os.path.join(os.getcwd(), file_name.split("_")[
                                         0], url[1].split("/")[-1])
            outfile = open(path_to_write, 'wb')
            outfile.write(resp.content)
            outfile.close()
            print("Done downloading {} of {}".format(url[0]+1, len(urls)))
        except:
            print("Failed to download url number {}".format(url[0]))
    t1 = time.time()
    print("Done with download, job took {} seconds".format(t1-t0))


def get_urls(image_tag, max_count):
    flickr = FlickrAPI(key, secret)
    photos = flickr.walk(
        text=image_tag,
        tag_mode='all',
        tags=image_tag,
        extras='url_o',
        per_page=50,
        sort='relevance'
    )
    count = 0
    urls = []
    for photo in photos:
        if count < max_count:
            count = count+1
            try:
                url = photo.get('url_o')
                urls.append(url)
                print(url)
            except:
                url
        else:
            break
    urls = pd.Series(urls)
    file_name = image_tag + "_urls.csv"
    urls.to_csv(file_name)
    return file_name


def main():
    tag = sys.argv[1]
    max_count = int(sys.argv[2])
    file_name = get_urls(tag, max_count)
    # get_urls(tag, max_count)
    file_name = 'clouds_urls.csv'
    put_images(file_name)


if __name__ == '__main__':
    main()
