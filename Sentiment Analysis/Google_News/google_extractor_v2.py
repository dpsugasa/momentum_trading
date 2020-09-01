
#%%
import asyncio
from GoogleNews import GoogleNews

class GoogleApi:
    def __init__(self):
        self._api = self._get_api()
def _get_news_for_word(self, word, top):
        self._api.search(word)
        text = self._api.result()
        news = list(map(lambda x: self._clean(x['title']) + ' @ (' + x['date'] + ')', text))[:top]
        return news
async def get_news_for_locations(self, target_locations, top=3):
        final = {}
        for location in target_locations:
            self._api.search(location)
            text = self._api.result()
            news = list(map(lambda x: self._clean(x['title']) + ' @ (' + x['date'] + ')', text))[:top]
            final[location] = news
        return final
def _get_api(self):
        return GoogleNews(period="1w")
async def get_news_for_words(self, target_words, top=3):
        text = {}
        for word in target_words:
            ts = self._get_news_for_word(word, top)
            text[word] = ts
        return text
async def get_topic_headlines(self, topics, top=5):
        topic_news = {}
        for topic in topics:
            try:
                self._api.search(topic)
                topic_headlines = self._api.result()
                topic_news[topic] = list(
                    map(lambda x: self._clean(x['title']) + ' @ (' + x['date'] + ')', topic_headlines))[:top]
            except Exception as e:
                print(topic)
                print(e)
                topic_news[topic] = []
        return topic_news
async def get_top_news(self):
        return self._api.top_news()
async def main():
    """
    Main method, call the appropriate methods
    :return:
    """
    google_api = GoogleApi()
    trends = google_api.get_news_for_locations(['United States', 'United Kingdom', 'Worldwide'])
    words = google_api.get_topic_headlines(['Coronavirus'])
    users = google_api.get_news_for_words(['bloomberg', 'cnbc'], 2)
    trends_text, words_text, news_users = await asyncio.gather(trends, words, users)
    print('LOCATION NEWS')
    for location, news in trends_text.items():
        print(f'Location: {location}  \n {news}')
    print('\nWORD NEWS')
    for word, news in words_text.items():
        print(f'Word: {word}  \n {news}')
    print('\nUSER NEWS')
    for user, news in news_users.items():
        print(f'User: {user}  \n {news}')
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait([main()]))