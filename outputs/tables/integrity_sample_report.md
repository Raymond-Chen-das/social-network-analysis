# Posts.xml Integrity Report — Sample Pass

- **File**: `C:\Users\raymo\Desktop\SCU work\社群網路分析\final_report\Posts.xml`
- **Size**: 103,933,385,042 bytes (96.80 GB)
- **XML header present**: False
- **`<posts` opening present**: True
- **`</posts>` closing present**: True

## File head (first 2 KB)

```xml
﻿<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="38779" PostTypeId="1" AcceptedAnswerId="40472" CreationDate="2008-09-02T03:41:06.880" Score="6" ViewCount="6282" Body="&lt;p&gt;I have a wcf application hosted in a windows service running a local windows account. Do I need to set an SPN for this account? If so, what's the protocol the SPN needs to be set under? I know how to do this for services over HTTP, but have never done it for net.tcp.&lt;/p&gt;&#xA;" OwnerUserId="781" OwnerDisplayName="Esteban" LastEditorUserId="1116" LastEditorDisplayName="John Nolan" LastEditDate="2008-09-02T08:49:19.323" LastActivityDate="2013-06-24T17:03:55.833" Title="What SPN do I need to set for a net.tcp service?" Tags="|wcf|security|spn|" AnswerCount="2" CommentCount="0" FavoriteCount="0" ContentLicense="CC BY-SA 2.5" />
  <row Id="38781" PostTypeId="2" ParentId="23930" CreationDate="2008-09-02T03:44:26.013" Score="3" Body="&lt;p&gt;Agda 2: Functional, dependently typed.&lt;/p&gt;&#xA;&#xA;&lt;pre&gt;&lt;code&gt;data Nat = zero | suc (m::Nat)&#xA;&#xA;add (m::Nat) (n::Nat) :: Nat&#xA; = case m of&#xA;     (zero ) -&amp;gt; n&#xA;     (suc p) -&amp;gt; suc (add p n)&#xA;&#xA;mul (m::Nat) (n::Nat)::Nat&#xA;   = case m of&#xA;      (zero ) -&amp;gt; zero&#xA;      (suc p) -&amp;gt; add n (mul p n)&#xA;&#xA;factorial (n::Nat)::Nat &#xA; = case n of&#xA;    (zero ) -&amp;gt; suc zero&#xA;    (suc p) -&amp;gt; mul n (factorial p)&#xA;&lt;/code&gt;&lt;/pre&gt;&#xA;" OwnerUserId="3434" LastEditorUserId="3434" LastEditorDisplayName="Apocalisp" LastEditDate="2008-09-18T22:09:07.470" LastActivityDate="2008-09-18T22:09:07.470" CommentCount="0" CommunityOwnedDate="2008-09-19T07:14:59.840" ContentLicense="CC BY-SA 2.5" />
  <row Id="38784" PostTypeId="1" AcceptedAnswerId="41285" CreationDate="2008-09-02T03:49:17.920" Score="7" ViewCount="3474" Body="&lt;p&gt;I use &lt;strong&gt;Delphi&lt;/strong&gt; for many years, and although I have now moved on to Visual Studio I still fondly remember numbered bookmarks (&lt;kbd&gt;CTRL&lt;/kb
```

## File tail (last 4 KB)

```xml
A;" OwnerUserId="518879" LastActivityDate="2024-03-24T16:45:41.180" CommentCount="0" ContentLicense="CC BY-SA 4.0" />
  <row Id="78253173" PostTypeId="2" ParentId="78218494" CreationDate="2024-03-31T23:56:29.537" Score="0" Body="&lt;p&gt;In the Delphi Programming Language, you can use the &lt;code&gt;StringReplace&lt;/code&gt; function from the &lt;code&gt;SysUtils&lt;/code&gt; unit to replace multiple spaces with a single space. However, &lt;code&gt;StringReplace&lt;/code&gt; only works on one occurrence at a time. To remove all instances of multiple spaces without looping, you can use a regular expression with the &lt;code&gt;TRegEx&lt;/code&gt; class from the &lt;code&gt;System.RegularExpressions&lt;/code&gt; unit.&lt;/p&gt;&#xA;&lt;p&gt;Here's a simple way to do it in one line:&lt;/p&gt;&#xA;&lt;pre&gt;&lt;code&gt;uses System.RegularExpressions;&#xA;&#xA;// ...&#xA;&#xA;ResultString := TRegEx.Replace(InputString, '\s+', ' ');&#xA;&lt;/code&gt;&lt;/pre&gt;&#xA;&lt;p&gt;This line of code will replace all sequences of one or more whitespace characters (\s+) in InputString with a single space, and store the result in ResultString. Remember to add &lt;code&gt;System.RegularExpressions&lt;/code&gt; to your uses clause to access the TRegEx class.&lt;/p&gt;&#xA;&lt;p&gt;Thank you for taking the time to view my post. I appreciate your attention and assistance!&lt;/p&gt;&#xA;" OwnerUserId="23498401" LastEditorUserId="2908017" LastEditDate="2024-04-01T01:59:09.660" LastActivityDate="2024-04-01T01:59:09.660" CommentCount="0" ContentLicense="CC BY-SA 4.0" />
  <row Id="78253175" PostTypeId="2" ParentId="24152351" CreationDate="2024-03-31T23:56:48.760" Score="0" Body="&lt;p&gt;Make the code in line 2 to:&#xA;int i = 0.00&lt;/p&gt;&#xA;&lt;p&gt;As java needs a value beforehand when you create the variable.&#xA;I hope it works fine after that:)&lt;/p&gt;&#xA;" OwnerUserId="23894764" LastActivityDate="2024-03-31T23:56:48.760" CommentCount="1" ContentLicense="CC BY-SA 4.0" />
  <row Id="78253176" PostTypeId="2" ParentId="78252549" CreationDate="2024-03-31T23:59:19.670" Score="0" Body="&lt;p&gt;This question is incomplete. We need to know what you use for styling in order to give you more guidance.&lt;/p&gt;&#xA;&lt;p&gt;If you use css for it, you can have a look at the focus pseudo-class. It is generally triggered, when the user clicks or taps on an element or selects it with the keyboard's Tab key. &lt;a href=&quot;https://developer.mozilla.org/en-US/docs/Web/CSS/:focus&quot; rel=&quot;nofollow noreferrer&quot;&gt;:focus&lt;/a&gt;&lt;/p&gt;&#xA;&lt;p&gt;&lt;div class=&quot;snippet&quot; data-lang=&quot;js&quot; data-hide=&quot;false&quot; data-console=&quot;true&quot; data-babel=&quot;false&quot;&gt;&#xD;&#xA;&lt;div class=&quot;snippet-code&quot;&gt;&#xD;&#xA;&lt;pre class=&quot;snippet-code-css lang-css prettyprint-override&quot;&gt;&lt;code&gt;label {&#xA;  display: block;&#xA;  margin-top: 1em;&#xA;}&#xA;&#xA;input:focus {&#xA;  background-color: lightblue;&#xA;}&#xA;&#xA;select:focus {&#xA;  background-color: ivory;&#xA;}&lt;/code&gt;&lt;/pre&gt;&#xD;&#xA;&lt;pre class=&quot;snippet-code-html lang-html prettyprint-override&quot;&gt;&lt;code&gt;&amp;lt;form&amp;gt;&#xA;  &amp;lt;p&amp;gt;Which flavor would you like to order?&amp;lt;/p&amp;gt;&#xA;  &amp;lt;label&amp;gt;Full Name: &amp;lt;input name=&quot;firstName&quot; type=&quot;text&quot; /&amp;gt;&amp;lt;/label&amp;gt;&#xA;  &amp;lt;label&#xA;    &amp;gt;Flavor:&#xA;    &amp;lt;select name=&quot;flavor&quot;&amp;gt;&#xA;      &amp;lt;option&amp;gt;Cherry&amp;lt;/option&amp;gt;&#xA;      &amp;lt;option&amp;gt;Green Tea&amp;lt;/option&amp;gt;&#xA;      &amp;lt;option&amp;gt;Moose Tracks&amp;lt;/option&amp;gt;&#xA;      &amp;lt;option&amp;gt;Mint Chip&amp;lt;/option&amp;gt;&#xA;    &amp;lt;/select&amp;gt;&#xA;  &amp;lt;/label&amp;gt;&#xA;&amp;lt;/form&amp;gt;&lt;/code&gt;&lt;/pre&gt;&#xD;&#xA;&lt;/div&gt;&#xD;&#xA;&lt;/div&gt;&#xD;&#xA;&lt;/p&gt;&#xA;" OwnerUserId="21338003" LastActivityDate="2024-03-31T23:59:19.670" CommentCount="0" ContentLicense="CC BY-SA 4.0" />
</posts>
```

## Sample scan (200,000 rows)

- Scan time: 7.4 s (26,854 rows/s)
- Date range in sample: 2008-07-31T21:42:52.667 → 2008-12-07T22:41:56.727

### PostTypeId distribution

| PostTypeId | Count | % |
|---|---:|---:|
| 2 — Answer | 158,011 | 79.01% |
| 1 — Question | 41,989 | 20.99% |

### Field presence (attributes on `<row>`)

| Attribute | Rows present | % |
|---|---:|---:|
| Id | 200,000 | 100.00% |
| PostTypeId | 200,000 | 100.00% |
| CreationDate | 200,000 | 100.00% |
| Score | 200,000 | 100.00% |
| Body | 200,000 | 100.00% |
| LastActivityDate | 200,000 | 100.00% |
| CommentCount | 200,000 | 100.00% |
| ContentLicense | 200,000 | 100.00% |
| OwnerUserId | 190,369 | 95.18% |
| OwnerDisplayName | 183,833 | 91.92% |
| ParentId | 158,011 | 79.01% |
| LastEditDate | 75,021 | 37.51% |
| LastEditorUserId | 73,343 | 36.67% |
| LastEditorDisplayName | 45,532 | 22.77% |
| Tags | 41,990 | 21.00% |
| ViewCount | 41,989 | 20.99% |
| Title | 41,989 | 20.99% |
| AnswerCount | 41,989 | 20.99% |
| AcceptedAnswerId | 32,141 | 16.07% |
| FavoriteCount | 28,238 | 14.12% |
| CommunityOwnedDate | 9,450 | 4.72% |
| ClosedDate | 3,991 | 2.00% |

### Question-specific checks

- Questions in sample: **41,989**
- Questions with Tags field: 41,989 (100.00%)
- Tag format `<tag1><tag2>`: 0
- Tag format `tag1|tag2|...`: 41,989
- Tag format other / unparseable: 0

### Deleted-user proxy (missing OwnerUserId)

- Rows missing OwnerUserId: 9,631 (4.82%)
