---
layout: post
title:  "closure for var in javascript"
date:   2016-10-24
categories: javascript
---

## clojure
{% highlight shell %}
(function (a) {
  return function () {
    console.log(a);
  }
})("arg");
{% endhighlight %}

## bind
{% highlight shell %}
(function (a) { console.log(a); }).bind("arg");
{% endhighlight %}
