#!/usr/bin/env python3.7
from base64 import b64decode, b64encode
import html_text
import numpy as np
from lxml import etree, html
import matplotlib.pyplot as plt
import os
import urllib.parse
import networkx as nx


def urljoin(base, url):
    try:
        return urllib.parse.urljoin(base, url)
    except:
        return url


class HTML:
    def __init__(self, element, html_bytes, tree):
        self.element = element
        self.bytes = html_bytes
        self.tree = tree

    def text(self):
        return html_text.extract_text(self.tree)

    def urls(self):
        doc_url = b64decode(self.element.find('docURL').text).decode("cp1251")
        res = [
            urljoin(doc_url, link_url)
            for _, _, link_url, _ in self.tree.iterlinks()
        ]
        res.insert(0, doc_url)
        return res


class Document:
    def __init__(self, element):
        self.element = element

    def html(self):
        html_bytes = self.content()
        return HTML(
            self.element,
            html_bytes,
            html.document_fromstring(html_bytes, parser=html.HTMLParser(encoding='cp1251')),
        )

    def content(self):
        return b64decode(self.element.find('content').text)


def iter_document_content(source):
    """

    :rtype: Iterator[:class:`xml.etree.Element`]
    """

    for _, element in etree.iterparse(source, events=('end',), tag='document'):
        try:
            yield Document(element)
        finally:
            del element.getparent()[0]


if __name__ == "__main__":
    byweb_for_course = os.path.join(os.path.dirname(__file__), 'data/byweb_for_course')
    from multiprocessing import Pool, freeze_support, RLock
    from tqdm import tqdm


    def partition_texts_base64(index):
        return os.path.join(byweb_for_course, f"texts.{index}.base64")


    def partition_url_base64(index):
        return os.path.join(byweb_for_course, f"urls.base64.{index}.csv")


    def partition_xml(index):
        return os.path.join(byweb_for_course, f"byweb.{index}.xml")


    def byte_lengths_csv():
        return os.path.join(byweb_for_course, f"byte_lengths.csv")


    def word_lengths_csv():
        return os.path.join(byweb_for_course, f"word_lengths.csv")


    def html_text_ratios_csv():
        return os.path.join(byweb_for_course, f"html_text_ratios.csv")


    def file_documents_number(index):
        return sum(1 for _ in tqdm(iter_document_content(partition_xml(index)), position=index))


    def extract_texts(index):
        with open(partition_texts_base64(index), "w") as output:
            for document in tqdm(iter_document_content(partition_xml(index)), position=index):
                print(b64encode(document.html().text().encode("utf-8")).decode(), file=output)


    def extract_urls(index):
        with open(partition_url_base64(index), "w") as output:
            for document in tqdm(iter_document_content(partition_xml(index)), position=index):
                print(",".join(b64encode(url.encode("utf-8")).decode() for url in document.html().urls()), file=output)


    def partition_byte_lengths(index):
        with open(partition_texts_base64(index)) as input:
            return [len(b64decode(text_base64).decode().encode('cp1251', 'replace')) for text_base64 in tqdm(input)]


    def partition_word_lengths(index):
        with open(partition_texts_base64(index)) as input:
            return [len(b64decode(text_base64).split()) for text_base64 in tqdm(input)]


    def partition_html_text_ratios(index):
        with open(partition_texts_base64(index)) as texts_base64:
            return [
                len(b64decode(text_base64).decode()) / len(document.content().decode('cp1251'))
                for text_base64, document in tqdm(zip(texts_base64, iter_document_content(partition_xml(index))))
            ]


    partitions = range(10)


    def top_n_url(n):
        dct = {}
        for part in partitions:
            with open(partition_url_base64(part)) as table:
                for line in table:
                    lst = line.split(",")
                    dct[lst[0]] = 0
            print(part)
        for part in partitions:
            with open(partition_url_base64(part)) as table:
                for line in table:
                    lst = line.split(",")
                    for url in lst:
                        if url in dct.keys():
                            dct[url] += 1
            print(part)
        urls = list(dct.keys())
        urls.sort(key=lambda x: dct[x], reverse=True)
        print("====")
        print(dct[urls[0]])
        print("====")
        return urls[:n]


    def make_table(nodes):
        g = nx.DiGraph()

        def get_index(url):
            try:
                return nodes.index(url)
            except ValueError:
                return None

        for part in partitions:
            with open(partition_url_base64(part)) as table:
                for line in table:
                    lst = line.split(",")
                    out_node = get_index(lst[0])
                    if out_node is not None:
                        for url in lst[1:]:
                            in_node = get_index(url)
                            if in_node is not None:
                                g.add_edge(in_node, out_node)
            print(part)
        return g

    def export_plots():
        byte_lengths_fig = plt.gcf()
        byte_lengths = np.genfromtxt(byte_lengths_csv())
        plt.axvline(byte_lengths.mean(), color='red', zorder=-1)
        plt.axvline(np.median(byte_lengths), color='green', zorder=-1)
        plot_equal_area_bins_hist(byte_lengths, range=(0, 50000))
        plt.title('Byte lengths')
        plt.legend([f"Mean={byte_lengths.mean():.0f}", f"Median={np.median(byte_lengths):.0f}"])
        plt.show()
        plt.draw()
        byte_lengths_fig.savefig('byte_lengths.png')

        word_lengths_fig = plt.gcf()
        word_lengths = np.genfromtxt(word_lengths_csv())
        plt.axvline(word_lengths.mean(), color='red', zorder=-1)
        plt.axvline(np.median(word_lengths), color='green', zorder=-1)
        plot_equal_area_bins_hist(word_lengths, range=(0, 7000))
        plt.title('Word lengths')
        plt.legend([f"Mean={word_lengths.mean():.1f}", f"Median={np.median(word_lengths):.0f}"])
        plt.show()
        plt.draw()
        word_lengths_fig.savefig('word_lengths.png')

        html_text_ratios_fig = plt.gcf()
        html_text_ratios = np.genfromtxt(html_text_ratios_csv())
        plt.axvline(html_text_ratios.mean(), color='red', zorder=-1)
        plt.axvline(np.median(html_text_ratios), color='green', zorder=-1)
        plot_equal_area_bins_hist(html_text_ratios)
        plt.title('Text to HTML ratios')
        plt.legend([f"Mean={html_text_ratios.mean():.2f}", f"Median={np.median(html_text_ratios):.2f}"])
        plt.show()
        plt.draw()
        html_text_ratios_fig.savefig('html_text_ratios.png')


    def plot_equal_area_bins_hist(x, bins=50, range=None):
        if range:
            x = x[(x >= range[0]) & (x <= range[1])]
        plt.yticks([])
        return plt.hist(
            x,
            np.interp(np.linspace(0, len(x), bins + 1), np.arange(len(x)), np.sort(x)),
            normed=True
        )


    def plot_hist_with_average(data, **args):
        plt.hist(data, bins=50, **args)
        plt.axvline(data.mean())
        plt.show()

    nx.write_graphml(make_table(top_n_url(300)), os.path.join(byweb_for_course, "300.graphml"))

    # freeze_support()
    # all_html_text_ratios = sum(Pool(len(partitions), initializer=tqdm.set_lock, initargs=(RLock(),)).map(
    #     partition_html_text_ratios,
    #     partitions
    # ), [])
    # with open(html_text_ratios_csv(), "w") as output:
    #     for ratio in all_html_text_ratios:
    #         print(ratio, file=output)
