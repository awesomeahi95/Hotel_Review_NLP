# -*- coding: utf-8 -*-
import scrapy


class HotelsSpider(scrapy.Spider):
    name = 'hotels'
    allowed_domains = ['www.tripadvisor.co.uk/Hotel_Review-g187051-d239658-Reviews-Hotel_Hilton_London_Gatwick_Airport-Crawley_West_Sussex_England.html']
    start_urls = ['https://www.tripadvisor.co.uk/Hotel_Review-g187051-d239658-Reviews-Hotel_Hilton_London_Gatwick_Airport-Crawley_West_Sussex_England.html/']

    def parse(self, response):
        for review in response.xpath('//div[@class="hotels-community-tab-common-Card__card--ihfZB hotels-community-tab-common-Card__section--4r93H"]'):
            yield {
                'score': review.xpath('.//div[@class="location-review-review-list-parts-RatingLine__bubbles--GcJvM"]/span').get(),
                'review': review.xpath('.//q[@class="location-review-review-list-parts-ExpandableReview__reviewText--gOmRC"]/span[1]/text()').get()
            }

#             //div[@class="location-review-review-list-parts-RatingLine__bubbles--GcJvM"]/span


# //div[@class="location-review-review-list-parts-ExpandableReview__containerStyles--1G0AE"]