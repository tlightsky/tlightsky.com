---
layout: post
title:  "Laravel在写新controller的时候找不到的问题"
date:   2016-05-12
categories: laravel controller
---

这个问题之前有遇到过，是通过更新artisan解决的：

{% highlight shell %}
./artisan dump-auto
{% endhighlight %}

但是，再之前也遇到过这次遇到的问题，就是artisan的命令不好使了
这种情况下，要用composer的才能更新这里的缓存


{% highlight shell %}
composer dump-autoload
{% endhighlight %}
